#!/bin/sh

rm -rf build/
mkdir build/
node rustdoc.js
cp -rp js build/
./node_modules/.bin/node-sass scss/main.scss build/main.css
