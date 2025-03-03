// Test to ensure that we can link to the current crate by using its name.

#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "bar"]

//@ has 'bar/index.html'
//@ has - '//*[@class="docblock"]//a[@href="index.html"]' 'bar'
//! [`bar`]

pub mod foo {
    //@ has 'bar/foo/fn.tadam.html'
    //@ has - '//*[@class="docblock"]//a[@href="../index.html"]' 'bar'
    /// [`bar`]
    pub fn tadam() {}
}
