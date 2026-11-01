// This test ensures that the intra-doc link from reexported deprecated attribute note
// are resolved where they are declared.

#![deny(rustdoc::broken_intra_doc_links)]

#[doc(inline)]
pub use bar::sql_function_proc as sql_function;

pub fn define_sql_function() {}

pub mod bar {
    #[deprecated(note = "Use [`define_sql_function`] instead")]
    //~^ ERROR: unresolved link
    //~| ERROR: unresolved link
    pub fn sql_function_proc() {}
}

// From here, this is a regression test for <https://github.com/rust-lang/rust/issues/151411>.
pub use fuzz_test_helpers::*;

/// A type referenced in the deprecation note.
pub struct Env;

impl Env {
    pub fn try_invoke(&self) {}
}

mod fuzz_test_helpers {
    #[deprecated(note = "use [Env::try_invoke] instead")]
    //~^ ERROR: unresolved link
    //~| ERROR: unresolved link
    pub fn fuzz_catch_panic() {}
}
