//@ check-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod inner {
    pub trait TraitFoo {
        fn foo(&self) -> u8;
    }
    pub trait TraitBar {
        fn bar(&self) -> u8;
    }

    impl TraitFoo for u8 {
        fn foo(&self) -> u8 { 0 }
    }
    impl TraitBar for u8 {
        fn bar(&self) -> u8 { 1 }
    }
}

trait Trait {
    fn foo(&self) -> u8;
    fn bar(&self) -> u8;
}

impl Trait for u8 {
    reuse inner::TraitFoo::*;
    reuse inner::TraitBar::*;
}

fn main() {
    let u = 0u8;
    u.foo();
    u.bar();
}
