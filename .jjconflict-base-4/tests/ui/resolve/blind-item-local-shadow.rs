//@ run-pass

#![allow(dead_code)]
#![allow(unused_imports)]
mod bar {
    pub fn foo() -> bool { true }
}

fn main() {
    let foo = || false;
    use bar::foo;
    assert_eq!(foo(), false);
}
