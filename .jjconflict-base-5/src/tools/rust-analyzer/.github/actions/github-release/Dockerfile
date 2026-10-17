FROM node:24.16-slim@sha256:ca520832af80fa37a57c14077ed0fcdd83b5aefccc356059fdc3a9a05b78ae1f

COPY . /action
WORKDIR /action

RUN npm install --production

ENTRYPOINT ["node", "/action/main.js"]
