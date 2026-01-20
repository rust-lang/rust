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
