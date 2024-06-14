# chatbot_test
**description**  
> Transformer을 활용한 간단한 챗봇 구현  
  
## RUN
```python main.py```  

### 실행시 주의 사항
1. 데이터 주소 : [ChatbotData.csv](https://drive.google.com/file/d/1muGP2mPof8hXEdcBECl0Qh6PWmf6m1bb/view?usp=sharing)
2. epoch 수 : main.py의 23번째 ```EPOCHS = nn``` 의 수를 변동하여 학습 수를 핸들링 합니다. 다만, 개인적인 테스트 결과 30 이상으로는 학습이 진행되지 않고 overfitting 문제가 발생하니 이 점 유의해주시길 바랍니다.
3. 미구현 사항 : 단순 테스트용이기 때문에 모델 저장까지 진행하지 않았습니다. 해당 사항은 추후 실 챗봇 서비스 배포시 구현하여 운영 관리할 예정이며 해당 레포는 테스트용으로 훈련과 동시에 챗봇 사이트 제공까지 진행되어 있습니다.

## Environment  
+ python version : 3.11  