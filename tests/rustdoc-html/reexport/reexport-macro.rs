// Ensure that macros are correctly reexported and that they get both the comment from the
// `pub use` and from the macro.

#![crate_name = "foo"]

//@ has 'foo/macro.foo.html'
//@ !has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'x y'
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'y'
#[macro_use]
mod my_module {
    /// y
    #[macro_export]
    macro_rules! foo {
        () => ();
    }
}

//@ has 'foo/another_mod/macro.bar.html'
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'x y'
pub mod another_mod {
    /// x
    pub use crate::foo as bar;
}
