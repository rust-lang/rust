#![crate_name = "foo"]

// Check that default trait items that are impossible to satisfy

pub trait Foo {
    fn needs_sized(&self)
    where
        Self: Sized,
    {}

    fn no_needs_sized(&self) {}
}

//@ !has foo/struct.Bar.html '//*[@id="method.needs_sized"]//h4[@class="code-header"]' \
// "fn needs_sized"
//@ has foo/struct.Bar.html '//*[@id="method.no_needs_sized"]//h4[@class="code-header"]' \
// "fn no_needs_sized"
pub struct Bar([u8]);

impl Foo for Bar {}

//@ has foo/struct.Generic.html '//*[@id="method.needs_sized"]//h4[@class="code-header"]' \
// "fn needs_sized"
//@ has foo/struct.Generic.html '//*[@id="method.no_needs_sized"]//h4[@class="code-header"]' \
// "fn no_needs_sized"
pub struct Generic<T: ?Sized>(T);

impl<T: ?Sized> Foo for Generic<T> {}
