#![allow(dead_code)]
#![allow(unused)]
//@ run-rustfix

mod m {
    pub(crate) struct S {
        pub(crate) visible: u64,
        hidden: u64,
    }

    impl S {
        pub(crate) fn new() -> Self {
            loop {}
        }
    }
}

fn main() {
    let m::S {
        //~^ ERROR pattern requires `..` due to inaccessible fields
        visible,
    } = m::S::new();
}
