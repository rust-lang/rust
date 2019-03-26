#![feature(cfg_if)]

macro_rules! cfg_if {
    (()) => {
        mod foo {
            fn foo() {}
        }
    }
}

cfg_if!{}

#[test]
fn it_works() {
    foo::foo();
}
