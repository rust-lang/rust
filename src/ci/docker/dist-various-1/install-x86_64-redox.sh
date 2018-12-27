#!/usr/bin/env bash
# ignore-tidy-linelength

set -ex

apt-get update
apt-get install -y --no-install-recommends software-properties-common apt-transport-https

apt-key adv --batch --yes --keyserver keyserver.ubuntu.com --recv-keys AA12E97F0881517F
add-apt-repository -y 'deb https://static.redox-os.org/toolchain/apt /'

apt-get update
apt-get install -y x86-64-unknown-redox-gcc
