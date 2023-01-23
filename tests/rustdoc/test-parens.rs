#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' "_: &(dyn ToString + 'static)"
pub fn foo(_: &(ToString + 'static)) {}
