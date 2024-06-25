#![crate_name = "foo"]

//@ has foo/fn.foo.html
//@ has - '//pre[@class="rust item-decl"]' "_: &(dyn ToString + 'static)"
pub fn foo(_: &(ToString + 'static)) {}
