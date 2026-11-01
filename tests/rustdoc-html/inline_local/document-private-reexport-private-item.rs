// This test ensures that we don't inline private imports automatically (and thus don't
// generate HTML files for them).

//@ compile-flags: --document-private-items

#![crate_name = "foo"]

//@ has foo/index.html
//@ has - '//*[@id="reexport.foo1"]/code' 'use foo as foo1;'
//@ has - '//*[@id="reexport.Bar1"]/code' 'pub(crate) use Bar as Bar1;'
//@ has foo/fn.foo.html
//@ has foo/struct.Bar.html
//@ !has foo/fn.foo1.html
//@ !has foo/struct.Bar1.html
fn foo() {}
struct Bar;

use foo as foo1;
pub(crate) use Bar as Bar1;
