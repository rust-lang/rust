// run-rustfix

#![allow(dead_code)]

struct SizedStruct(i32);
struct UnsizedStruct([i32]);
struct BigStruct([i32; 10000]);

/// The following should trigger the lint
mod should_trigger {
    use super::SizedStruct;

    struct StructWithVecBox {
        sized_type: Vec<Box<SizedStruct>>,
    }

    struct A(Vec<Box<SizedStruct>>);
    struct B(Vec<Vec<Box<(u32)>>>);
}

/// The following should not trigger the lint
mod should_not_trigger {
    use super::{BigStruct, UnsizedStruct};

    struct C(Vec<Box<UnsizedStruct>>);
    struct D(Vec<Box<BigStruct>>);

    struct StructWithVecBoxButItsUnsized {
        unsized_type: Vec<Box<UnsizedStruct>>,
    }

    struct TraitVec<T: ?Sized> {
        // Regression test for #3720. This was causing an ICE.
        inner: Vec<Box<T>>,
    }
}

mod inner_mod {
    mod inner {
        pub struct S;
    }

    mod inner2 {
        use super::inner::S;

        pub fn f() -> Vec<Box<S>> {
            vec![]
        }
    }
}

fn main() {}
