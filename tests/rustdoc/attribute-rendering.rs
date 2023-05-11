#![crate_name = "foo"]

// @has 'foo/fn.f.html'
// @has - //*[@'class="rust item-decl"]' '#[export_name = "f"] pub fn f()'
#[export_name = "\
f"]
pub fn f() {}
