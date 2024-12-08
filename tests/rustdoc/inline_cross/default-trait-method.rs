//@ aux-build:default-trait-method.rs

extern crate foo;

//@ has default_trait_method/trait.Item.html
//@ has - '//*[@id="tymethod.foo"]' 'fn foo()'
//@ !has - '//*[@id="tymethod.foo"]' 'default fn foo()'
//@ has - '//*[@id="tymethod.bar"]' 'fn bar()'
//@ !has - '//*[@id="tymethod.bar"]' 'default fn bar()'
//@ has - '//*[@id="method.baz"]' 'fn baz()'
//@ !has - '//*[@id="method.baz"]' 'default fn baz()'
pub use foo::Item;

//@ has default_trait_method/struct.Foo.html
//@ has - '//*[@id="method.foo"]' 'default fn foo()'
//@ has - '//*[@id="method.bar"]' 'fn bar()'
//@ !has - '//*[@id="method.bar"]' 'default fn bar()'
//@ has - '//*[@id="method.baz"]' 'fn baz()'
//@ !has - '//*[@id="method.baz"]' 'default fn baz()'
pub use foo::Foo;
