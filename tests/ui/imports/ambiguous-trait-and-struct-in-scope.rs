//@ check-pass
//@ edition:2018
//
// Tests that when the primary declaration is not a trait but the ambiguous declaration is,
// the ambiguous trait is still recorded among the module's traits. This is to make sure
// that we can report the `ambiguously_glob_import_trait` lint.
//
// Test case is from #159476.

mod module_1 {
    mod nested_1 {
        pub struct Foo;
    }

    pub use nested_1::Foo;
}

mod module_2 {
    // same name as the struct
    pub trait Foo: Sized {
        fn method(self) {}
    }
    impl Foo for i32 {}
}

mod module_3 {
    mod nested_3 {
        use super::*;
        use crate::module_2::*;
        fn weird() {
            1_i32.method(); //~ WARNING Use of ambiguously glob imported trait `Foo` [ambiguous_glob_imported_traits]
                            //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        }
    }

    use crate::module_1::*;
}

fn main() {}
