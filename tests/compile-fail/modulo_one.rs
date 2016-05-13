#![feature(plugin)]
#![plugin(clippy)]
#![deny(modulo_one)]
#![allow(no_effect, unnecessary_operation)]

fn main() {
    10 % 1; //~ERROR any number modulo 1 will be 0
    10 % 2;
}
