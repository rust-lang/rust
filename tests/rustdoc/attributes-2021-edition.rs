//@ edition: 2021
#![crate_name = "foo"]

//@ has foo/fn.f.html '//pre[@class="rust item-decl"]' '#[unsafe(no_mangle)]'
#[no_mangle]
pub extern "C" fn f() {}

//@ has foo/fn.g.html '//pre[@class="rust item-decl"]' '#[unsafe(export_name = "bar")]'
#[export_name = "bar"]
pub extern "C" fn g() {}

//@ has foo/fn.example.html '//pre[@class="rust item-decl"]' '#[unsafe(link_section = ".text")]'
#[link_section = ".text"]
pub extern "C" fn example() {}
