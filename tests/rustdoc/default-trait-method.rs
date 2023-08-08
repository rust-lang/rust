#![feature(min_specialization)]

// @has default_trait_method/trait.Item.html
pub trait Item {
    // @has - '//*[@id="tymethod.foo"]' 'fn foo()'
    // @!has - '//*[@id="tymethod.foo"]' 'default fn foo()'
    fn foo();

    // @has - '//*[@id="tymethod.bar"]' 'fn bar()'
    // @!has - '//*[@id="tymethod.bar"]' 'default fn bar()'
    fn bar();

    // @has - '//*[@id="tymethod.baz"]' 'unsafe fn baz()'
    // @!has - '//*[@id="tymethod.baz"]' 'default unsafe fn baz()'
    unsafe fn baz();

    // @has - '//*[@id="tymethod.quux"]' 'unsafe fn quux()'
    // @!has - '//*[@id="tymethod.quux"]' 'default unsafe fn quux()'
    unsafe fn quux();

    // @has - '//*[@id="method.xyzzy"]' 'fn xyzzy()'
    // @!has - '//*[@id="method.xyzzy"]' 'default fn xyzzy()'
    fn xyzzy() {}
}

// @has default_trait_method/struct.Foo.html
pub struct Foo;
impl Item for Foo {
    // @has - '//*[@id="method.foo"]' 'default fn foo()'
    default fn foo() {}

    // @has - '//*[@id="method.bar"]' 'fn bar()'
    // @!has - '//*[@id="method.bar"]' 'default fn bar()'
    fn bar() {}

    // @has - '//*[@id="method.baz"]' 'default unsafe fn baz()'
    default unsafe fn baz() {}

    // @has - '//*[@id="method.quux"]' 'unsafe fn quux()'
    // @!has - '//*[@id="method.quux"]' 'default unsafe fn quux()'
    unsafe fn quux() {}

    // @has - '//*[@id="method.xyzzy"]' 'fn xyzzy()'
    // @!has - '//*[@id="method.xyzzy"]' 'default fn xyzzy()'
}
