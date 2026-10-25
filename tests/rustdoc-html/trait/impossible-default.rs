#![crate_name = "foo"]

// Check that default trait items that are impossible to satisfy are not
// listed.

pub trait Foo {
    fn needs_sized(&self)
    where
        Self: Sized,
    {}

    fn no_needs_sized(&self) {}
}

//@ has 'foo/struct.Bar.html'
// "fn needs_sized" should not be listed.
//@ count - '//*[@id="trait-implementations-list"]//h4[@class="code-header"]' 1
//@ has - '//*[@id="trait-implementations-list"]//h4[@class="code-header"]' "fn no_needs_sized"
pub struct Bar([u8]);

impl Foo for Bar {}

//@ has 'foo/struct.Generic.html'
//@ has - '//*[@id="method.needs_sized"]//h4[@class="code-header"]' "fn needs_sized"
//@ has - '//*[@id="method.no_needs_sized"]//h4[@class="code-header"]' "fn no_needs_sized"
pub struct Generic<T: ?Sized>(T);

impl<T: ?Sized> Foo for Generic<T> {}
