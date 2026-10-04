#![crate_type="lib"]

pub struct Foo;

impl Foo {
    //@ has const/type.Foo.html '//*[@id="method.new"]//h4[@class="code-header"]' 'const unsafe fn new'
    pub const unsafe fn new() -> Foo {
        Foo
    }
}
