#![feature(final_associated_functions)]

//@ has final_trait_method/trait.Item.html
pub trait Item {
    //@ has - '//*[@id="method.foo"]' 'final fn foo()'
    //@ !has - '//*[@id="method.foo"]' 'default fn foo()'
    final fn foo() {}

    //@ has - '//*[@id="method.bar"]' 'fn bar()'
    //@ !has - '//*[@id="method.bar"]' 'default fn bar()'
    //@ !has - '//*[@id="method.bar"]' 'final fn bar()'
    fn bar() {}
}

//@ has final_trait_method/struct.Foo.html
pub struct Foo;
impl Item for Foo {
    //@ has - '//*[@id="method.bar"]' 'fn bar()'
    //@ !has - '//*[@id="method.bar"]' 'final fn bar()'
    //@ !has - '//*[@id="method.bar"]' 'default fn bar()'
    fn bar() {}
}
