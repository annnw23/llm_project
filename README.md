packages to install

```
pip install torchvision
pip install optuna
pip install gradio
```


maybe to install
```
pip install lime
pip install grad-cam
```


Notatka do zajęć 
- Metoda wyjaśnialności – Gradio - zaprezentować które piksele były najważniejsze dla modelu
- Optuna – na początku mocno zawęzić gardło i puścić optymalizację na noc, następnie stopniowo dopasowywać parametry
- Raporty regularnie zbierać i wrzucać podpisane na GitHub
- Przygotować końcową prezentację projektu
- Korzystać z Google Colab
- adam i sgd
- żeby on skakał po: liczba i rozpiętności kerneli, learning rate, bach size
- co jest dokładność co jest strata modeli
- W każdym podaniu trenowanie i testowanie i potem sprawdzenie metryki




[COLAB](https://colab.research.google.com/drive/1vk2aimVFx0hEzA33-by8t_c21GTNoFGw?usp=sharing#scrollTo=3--IcskcVOqC)

Nasz projekt polegał na analizie działania sieci neuronowych oraz optymalizacji ich parametrów dla zadania klasyfikacji obrazów ze zbioru MNIST. W pierwszej części wykorzystano bibliotekę Optuna do automatycznego doboru hiperparametrów konwolucyjnej sieci neuronowej CNN. Dla każdej wygenerowanej konfiguracji model był trenowany przez określoną liczbę epok, a po każdej epoce obliczana była dokładność na zbiorze walidacyjnym. Wyniki były przekazywane do Optuny, która na ich podstawie wybierała najbardziej obiecujące kombinacje parametrów do kolejnych prób. Po wyznaczeniu najlepszej konfiguracji model został ponownie wytrenowany na pełnym zbiorze treningowym i oceniony na zbiorze testowym. Dodatkowo zaimplementowano interfejs użytkownika w bibliotece Gradio, umożliwiający wgrywanie lub rysowanie cyfr i ich klasyfikację przez wytrenowany model. W celu zwiększenia interpretowalności wyników wykorzystano bibliotekę SHAP, która pozwala wizualizować wpływ poszczególnych pikseli obrazu na końcową decyzję sieci neuronowej.

Po napisaniu pierwszej wersji kodu (test.py) postanowiliśmy przetestować możliwie szeroki zakres hiperparametrów. Celem było określenie, które wartości mają największy wpływ na skuteczność modelu oraz zawężenie zakresów do dalszych testów.

<img width="633" height="281" alt="image" src="https://github.com/user-attachments/assets/f233d870-bc89-4b35-a0f7-aa8cbebc4de6" />

