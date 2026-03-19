#![deny(dead_code)]
#![deny(unused_pub_items_in_binary)]

pub fn g() {} //~ ERROR function `g` is never used

fn h() {}

#[allow(dead_code)]
fn f() {
    g();
    h();
}

fn main() {}
