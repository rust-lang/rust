FROM node:slim

COPY . /action
WORKDIR /action

RUN npm install --production

ENTRYPOINT ["node", "/action/main.js"]
