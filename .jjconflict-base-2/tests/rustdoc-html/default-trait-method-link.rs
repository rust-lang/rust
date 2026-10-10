#![crate_name = "foo"]

//@ has foo/trait.Foo.html '//a[@href="trait.Foo.html#tymethod.req"]' 'req'
//@ has foo/trait.Foo.html '//a[@href="trait.Foo.html#method.prov"]' 'prov'

/// Always make sure to implement [`req`], but you don't have to implement [`prov`].
///
/// [`req`]: Foo::req
/// [`prov`]: Foo::prov
pub trait Foo {
    /// Required
    fn req();
    /// Provided
    fn prov() {}
}
