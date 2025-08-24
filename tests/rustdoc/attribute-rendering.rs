#![crate_name = "foo"]

//@ has 'foo/fn.f.html'
//@ has - //*[@'class="code-attribute"]' '#[unsafe(export_name = "f")]'
//@ has - //*[@'class="rust item-decl"]' 'pub fn f()'
#[unsafe(export_name = "\
f")]
pub fn f() {}
