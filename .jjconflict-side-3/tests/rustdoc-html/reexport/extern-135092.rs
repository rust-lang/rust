// Test to make sure reexports of extern items are combined
// <https://github.com/rust-lang/rust/issues/135092>

#![crate_name = "foo"]

mod native {
    extern "C" {
        /// bar.
        pub fn bar();
    }

    /// baz.
    pub fn baz() {}
}

//@ has 'foo/fn.bar.html'
//@ has - '//div[@class="docblock"]' 'bar.'
//@ has - '//div[@class="docblock"]' 'foo'
/// foo
pub use native::bar;

//@ has 'foo/fn.baz.html'
//@ has - '//div[@class="docblock"]' 'baz.'
//@ has - '//div[@class="docblock"]' 'foo'
/// foo
pub use native::baz;
