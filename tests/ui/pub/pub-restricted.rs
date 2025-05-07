mod a {}

pub (a) fn afn() {} //~ ERROR incorrect visibility restriction
pub (b) fn bfn() {} //~ ERROR incorrect visibility restriction
pub (crate::a) fn cfn() {} //~ ERROR incorrect visibility restriction

pub fn privfn() {}
mod x {
    mod y {
        pub (in x) fn foo() {}
        pub (super) fn bar() {}
        pub (crate) fn qux() {}
    }
}

mod y {
    struct Foo {
        pub (crate) c: usize,
        pub (super) s: usize,
        valid_private: usize,
        pub (in y) valid_in_x: usize,
        pub (a) invalid: usize, //~ ERROR incorrect visibility restriction
        pub (in x) non_parent_invalid: usize, //~ ERROR visibilities can only be restricted
    }
}

fn main() {}

// test multichar names
mod xyz {}
pub (xyz) fn xyz() {} //~ ERROR incorrect visibility restriction
