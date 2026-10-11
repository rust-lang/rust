#![crate_name = "foo"]

// https://github.com/rust-lang/rust/issues/34473

mod second {
    pub struct SomeTypeWithLongName;
}

//@ has foo/index.html
//@ !hasraw - SomeTypeWithLongName
//@ has foo/type.SomeType.html
//@ !has foo/type.SomeTypeWithLongName.html
pub use second::{SomeTypeWithLongName as SomeType};
