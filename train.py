import os
import csv
import torch
import torch.optim as optim
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()  # Coloca o modelo em modo de treinamento
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()           # Zera os gradientes antes da iteração
        output = model(data)            # Forward pass: gera a previsão para os dados
        loss = F.nll_loss(output, target)  # Calcula a perda
        loss.backward()                 # Backpropagation: calcula os gradientes
        optimizer.step()                # Atualiza os parâmetros do modelo

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()  # Coloca o modelo em modo de avaliação
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Desativa o cálculo de gradientes para economizar memória
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Soma a perda do batch
            pred = output.argmax(dim=1, keepdim=True)  # Prediz a classe com maior probabilidade
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy

def main():
    import argparse
    from model import Net
    from data_loader import get_data_loaders

    # Argumentos de linha de comando
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Cria pastas para salvar resultados e checkpoints, se não existirem
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Define o arquivo CSV onde as métricas serão salvas
    csv_file = os.path.join("results", "metrics.csv")
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Test Loss", "Test Accuracy"])

    # Carrega os DataLoaders
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    # Inicializa o modelo e o envia para o dispositivo
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0  # Variável para guardar a melhor acurácia

    # Loop de treinamento e avaliação
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=args.log_interval)
        test_loss, accuracy = test(model, device, test_loader)

        # Salva as métricas de cada época no arquivo CSV
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, test_loss, accuracy])

        # Salva o modelo se a acurácia for a melhor até agora
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Melhor modelo salvo na época {epoch} com acurácia de {accuracy:.2f}%")

if __name__ == '__main__':
    main()
