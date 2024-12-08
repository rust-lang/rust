//@aux-build:proc_macro_derive.rs

#![allow(unused, clippy::duplicated_attributes)]
#![warn(clippy::useless_attribute)]
#![warn(unreachable_pub)]
#![feature(rustc_private)]

#[allow(dead_code)]
#[cfg_attr(clippy, allow(dead_code))]
#[rustfmt::skip]
#[allow(unused_imports)]
#[allow(unused_extern_crates)]
#[macro_use]
extern crate rustc_middle;

#[macro_use]
extern crate proc_macro_derive;

fn test_indented_attr() {
    #[allow(clippy::almost_swapped)]
    use std::collections::HashSet;

    let _ = HashSet::<u32>::default();
}

// don't lint on unused_import for `use` items
#[allow(unused_imports)]
use std::collections;

// don't lint on unused for `use` items
#[allow(unused)]
use std::option;

// don't lint on deprecated for `use` items
mod foo {
    #[deprecated]
    pub struct Bar;
}
#[allow(deprecated)]
pub use foo::Bar;

// This should not trigger the lint. There's lint level definitions inside the external derive
// that would trigger the useless_attribute lint.
#[derive(DeriveSomething)]
struct Baz;

// don't lint on unreachable_pub for `use` items
mod a {
    mod b {
        #[allow(dead_code)]
        #[allow(unreachable_pub)]
        pub struct C;
    }

    #[allow(unreachable_pub)]
    pub use self::b::C;
}

// don't lint on clippy::wildcard_imports for `use` items
#[allow(clippy::wildcard_imports)]
pub use std::io::prelude::*;

// don't lint on clippy::enum_glob_use for `use` items
#[allow(clippy::enum_glob_use)]
pub use std::cmp::Ordering::*;

// don't lint on clippy::redundant_pub_crate
mod c {
    #[allow(clippy::redundant_pub_crate)]
    pub(crate) struct S;
}

// https://github.com/rust-lang/rust-clippy/issues/7511
pub mod split {
    #[allow(clippy::module_name_repetitions)]
    pub use regex::SplitN;
}

// https://github.com/rust-lang/rust-clippy/issues/8768
#[allow(clippy::single_component_path_imports)]
use regex;

mod module {
    pub(crate) struct Struct;
}

#[rustfmt::skip]
#[allow(unused_import_braces)]
#[allow(unused_braces)]
use module::{Struct};

fn main() {
    test_indented_attr();
}

// Regression test for https://github.com/rust-lang/rust-clippy/issues/4467
#[allow(dead_code)]
use std::collections as puppy_doggy;

// Regression test for https://github.com/rust-lang/rust-clippy/issues/11595
pub mod hidden_glob_reexports {
    #![allow(unreachable_pub)]

    mod my_prelude {
        pub struct MyCoolTypeInternal;
        pub use MyCoolTypeInternal as MyCoolType;
    }

    mod my_uncool_type {
        pub(crate) struct MyUncoolType;
    }

    // This exports `MyCoolType`.
    pub use my_prelude::*;

    // This hides `my_prelude::MyCoolType`.
    #[allow(hidden_glob_reexports)]
    use my_uncool_type::MyUncoolType as MyCoolType;
}

// Regression test for https://github.com/rust-lang/rust-clippy/issues/10878
pub mod ambiguous_glob_exports {
    #![allow(unreachable_pub)]

    mod my_prelude {
        pub struct MyType;
    }

    mod my_type {
        pub struct MyType;
    }

    #[allow(ambiguous_glob_reexports)]
    pub use my_prelude::*;
    pub use my_type::*;
}
