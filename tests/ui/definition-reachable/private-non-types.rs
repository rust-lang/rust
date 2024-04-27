// Check that we don't require stability annotations for private modules,
// imports and fields that are accessible to opaque macros.

//@ check-pass

#![feature(decl_macro, staged_api)]
#![stable(feature = "test", since = "1.0.0")]

extern crate std as local_std;
use local_std::marker::Copy as LocalCopy;
mod private_mod {
    #[stable(feature = "test", since = "1.0.0")]
    pub struct A {
        pub(crate) f: i32,
    }
}

#[stable(feature = "test", since = "1.0.0")]
pub macro m() {}

fn main() {}
