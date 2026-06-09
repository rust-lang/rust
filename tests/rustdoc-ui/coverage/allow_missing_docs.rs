//@ compile-flags:-Z unstable-options --show-coverage
//@ check-pass

//! Make sure to have some docs on your crate root

#[allow(missing_docs)]
pub mod mod_foo {
    pub struct Bar;
}

/// This is a struct with an `#[allow(missing_docs)]`
pub struct AllowTheMissingDocs {
    #[allow(missing_docs)]
    pub empty_str: String,

    /// This has
    #[allow(missing_docs)]
    /// but also has documentation comments
    pub hello: usize,

    /// The doc id just to create a boilerplate comment
    pub doc_id: Vec<u8>,
}

/// A function that has a documentation
pub fn this_is_func() {}

#[allow(missing_docs)]
pub struct DemoStruct {
    something: usize,
}

#[allow(missing_docs)]
pub mod bar {
    #[warn(missing_docs)]
    pub struct Bar { //~ WARN
        pub f: u32, //~ WARN
    }

    pub struct NeedsNoDocs;
}
