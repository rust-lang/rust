#![crate_name = "foo"]

// @has foo/fn.f.html '//*[@class="rust fn"]' '#[no_mangle]'
#[no_mangle]
pub extern "C" fn f() {}

// @has foo/fn.g.html '//*[@class="rust fn"]' '#[export_name = "bar"]'
#[export_name = "bar"]
pub extern "C" fn g() {}

// @has foo/struct.Repr.html '//*[@class="docblock item-decl"]' '#[repr(C, align(8))]'
#[repr(C, align(8))]
pub struct Repr;
