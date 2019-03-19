#![feature(specialization)]

pub trait Item {
    fn foo();
    fn bar();
}

// @has default_trait_method/trait.Item.html
// @has - '//*[@id="method.foo"]' 'default fn foo()'
// @has - '//*[@id="method.bar"]' 'fn bar()'
// @!has - '//*[@id="method.bar"]' 'default fn bar()'
impl<T: ?Sized> Item for T {
    default fn foo() {}
    fn bar() {}
}
