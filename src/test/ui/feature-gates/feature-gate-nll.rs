// This is a test checking that if you do not opt into NLL then you
// should not get the effects of NLL applied to the test.

// Don't use 2018 edition, since that turns on NLL (migration mode).
// edition:2015

// Don't use compare-mode=nll, since that turns on NLL.
// ignore-compare-mode-nll


#![allow(dead_code)]

fn main() {
    let mut x = 33;

    let p = &x;
    x = 22; //~ ERROR cannot assign to `x` because it is borrowed [E0506]
}
