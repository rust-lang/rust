#![feature(associated_type_defaults)]

// @has issue_28478/trait.Bar.html
pub trait Bar {
    // @has - '//*[@id="associatedtype.Bar"]' 'type Bar = ()'
    // @has - '//*[@href="#associatedtype.Bar"]' 'Bar'
    type Bar = ();
    // @has - '//*[@id="associatedconstant.Baz"]' 'const Baz: usize'
    // @has - '//*[@href="#associatedconstant.Baz"]' 'Baz'
    const Baz: usize = 7;
    // @has - '//*[@id="tymethod.bar"]' 'fn bar'
    fn bar();
    // @has - '//*[@id="method.baz"]' 'fn baz'
    fn baz() { }
}

// @has issue_28478/struct.Foo.html
pub struct Foo;

impl Foo {
    // @has - '//*[@href="#method.foo"]' 'foo'
    pub fn foo() {}
}

impl Bar for Foo {
    // @has - '//*[@href="../issue_28478/trait.Bar.html#associatedtype.Bar"]' 'Bar'
    // @has - '//*[@href="../issue_28478/trait.Bar.html#associatedconstant.Baz"]' 'Baz'
    // @has - '//*[@href="../issue_28478/trait.Bar.html#tymethod.bar"]' 'bar'
    fn bar() {}
    // @has - '//*[@href="../issue_28478/trait.Bar.html#method.baz"]' 'baz'
}
