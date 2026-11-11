// We check that if `--document-private-items` is used, the link to the reexported item
// is generated.

//@ compile-flags: --document-private-items

#![crate_name = "foo"]

mod a {
    pub fn foo() {}
}

//@ has 'foo/fn.bar.html'
//@ has - '//*[@class="reexport-origin"]' 'Reexported from crate::a::foo'
//@ has - '//*[@class="reexport-origin"]/a[@href="a/fn.foo.html"]' 'crate::a::foo'
#[doc(inline)]
pub use crate::a::foo as bar;
