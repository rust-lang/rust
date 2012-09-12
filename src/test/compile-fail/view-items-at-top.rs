// xfail-test

extern mod std;

fn f() {
}

use std::net;    //~ ERROR view items must be declared at the top

fn main() {
}

