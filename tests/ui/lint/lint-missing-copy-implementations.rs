// See issue 19712

#![deny(missing_copy_implementations)]

mod inner {
    pub struct Foo { //~ ERROR type could implement `Copy`; consider adding `impl Copy`
        pub field: i32
    }
}

pub fn foo() -> inner::Foo {
    inner::Foo { field: 42 }
}

fn main() {}
