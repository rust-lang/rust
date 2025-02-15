#![allow(dead_code, clippy::extra_unused_lifetimes)]
#![warn(clippy::multiple_inherent_impl)]

struct MyStruct;

impl MyStruct {
    fn first() {}
}

impl MyStruct {
    //~^ multiple_inherent_impl

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
        //~^ multiple_inherent_impl

        fn third() {}
    }
}

use std::fmt;
impl fmt::Debug for MyStruct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MyStruct {{ }}")
    }
}

// issue #5772
struct WithArgs<T>(T);
impl WithArgs<u32> {
    fn f1() {}
}
impl WithArgs<u64> {
    fn f2() {}
}
impl WithArgs<u64> {
    //~^ multiple_inherent_impl

    fn f3() {}
}

// Ok, the struct is allowed to have multiple impls.
#[allow(clippy::multiple_inherent_impl)]
struct Allowed;
impl Allowed {}
impl Allowed {}
impl Allowed {}

struct AllowedImpl;
#[allow(clippy::multiple_inherent_impl)]
impl AllowedImpl {}
// Ok, the first block is skipped by this lint.
impl AllowedImpl {}

struct OneAllowedImpl;
impl OneAllowedImpl {}
#[allow(clippy::multiple_inherent_impl)]
impl OneAllowedImpl {}
impl OneAllowedImpl {} // Lint, only one of the three blocks is allowed.
//~^ multiple_inherent_impl

fn main() {}
