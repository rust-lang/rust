//! Test that overriding cfg_if with our own cfg_if macro does not break
//! anything.

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
