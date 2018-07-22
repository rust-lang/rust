node('master') {
  stage('Checkout') {
    checkout scm
    sh 'git submodule update --init --recursive'
  }

  stage('Build') {
      dir('build') {
        sh '../x.py build'
      }
  }
}


