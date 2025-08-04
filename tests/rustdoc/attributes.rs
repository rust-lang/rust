//@ edition: 2024
#![crate_name = "foo"]

//@ has foo/fn.f.html '//pre[@class="rust item-decl"]' '#[unsafe(no_mangle)]'
#[unsafe(no_mangle)]
pub extern "C" fn f() {}

//@ has foo/fn.g.html '//pre[@class="rust item-decl"]' '#[unsafe(export_name = "bar")]'
#[unsafe(export_name = "bar")]
pub extern "C" fn g() {}

//@ has foo/fn.example.html '//pre[@class="rust item-decl"]' '#[unsafe(link_section = ".text")]'
#[unsafe(link_section = ".text")]
pub extern "C" fn example() {}

//@ has foo/struct.Repr.html '//pre[@class="rust item-decl"]' '#[repr(C, align(8))]'
#[repr(C, align(8))]
pub struct Repr;
