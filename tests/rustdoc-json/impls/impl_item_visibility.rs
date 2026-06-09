pub struct Foo;

/// impl Foo priv
impl Foo {
    fn baz() {}
}
//@ !has '$.index[?(@.docs=="impl Foo priv")]'

/// impl Foo pub
impl Foo {
    pub fn qux() {}
}
//@ is '$.index[?(@.docs=="impl Foo pub")].visibility' '"default"'

/// impl Foo hidden
impl Foo {
    #[doc(hidden)]
    pub fn __quazl() {}
}
// FIXME(#111564): Is this the right behaviour?
//@ is '$.index[?(@.docs=="impl Foo hidden")].visibility' '"default"'
