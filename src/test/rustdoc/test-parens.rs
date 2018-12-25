#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - '//*[@class="rust fn"]' "_: &(dyn ToString + 'static)"
pub fn foo(_: &(ToString + 'static)) {}
