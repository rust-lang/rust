#![feature(plugin)]
#![plugin(clippy)]
#![deny(modulo_one)]

fn main() {
    10 % 1; //~ERROR Any number modulo 1 will be 0
    10 % 2;
}
