#![crate_name = "foo"]

//@ has 'foo/fn.f.html'
//@ has - //*[@'class="rust item-decl"]' '#[unsafe(export_name = "f")] pub fn f()'
#[unsafe(export_name = "\
f")]
pub fn f() {}
