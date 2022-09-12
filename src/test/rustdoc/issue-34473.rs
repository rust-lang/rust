#![crate_name = "foo"]

mod second {
    pub struct SomeTypeWithLongName;
}

// @has foo/index.html
// @!hasraw - SomeTypeWithLongName
// @has foo/struct.SomeType.html
// @!has foo/struct.SomeTypeWithLongName.html
pub use second::{SomeTypeWithLongName as SomeType};
