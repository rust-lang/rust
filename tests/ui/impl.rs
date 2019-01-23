#![allow(dead_code)]
#![warn(clippy::multiple_inherent_impl)]

struct MyStruct;

impl MyStruct {
    fn first() {}
}

impl MyStruct {
    fn second() {}
}

impl<'a> MyStruct {
    fn lifetimed() {}
}

mod submod {
    struct MyStruct;
    impl MyStruct {
        fn other() {}
    }

    impl super::MyStruct {
        fn third() {}
    }
}

use std::fmt;
impl fmt::Debug for MyStruct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MyStruct {{ }}")
    }
}

fn main() {}
