#![deny(dead_code)]
#![deny(dead_code_pub_in_binary)]

pub fn g() {} //~ ERROR function `g` is never used

fn h() {}

#[allow(dead_code)]
fn f() {
    g();
    h();
}

fn main() {}
