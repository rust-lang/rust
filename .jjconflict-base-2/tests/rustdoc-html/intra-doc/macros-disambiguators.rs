#![crate_name = "foo"]
#![deny(rustdoc::broken_intra_doc_links)]

//! [foo!()]
//@ has foo/index.html '//a[@href="macro.foo.html"]' 'foo!()'

//! [foo!{}]
//@ has - '//a[@href="macro.foo.html"]' 'foo!{}'

//! [foo![]](foo![])
//@ has - '//a[@href="macro.foo.html"]' 'foo![]'

//! [foo1](foo!())
//@ has - '//a[@href="macro.foo.html"]' 'foo1'

//! [foo2](foo!{})
//@ has - '//a[@href="macro.foo.html"]' 'foo2'

//! [foo3](foo![])
//@ has - '//a[@href="macro.foo.html"]' 'foo3'

#[macro_export]
macro_rules! foo {
    () => {};
}
