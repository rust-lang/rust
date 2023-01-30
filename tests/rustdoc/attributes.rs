#![crate_name = "foo"]

// @has foo/fn.f.html '//div[@class="item-decl"]/pre[@class="rust"]' '#[no_mangle]'
#[no_mangle]
pub extern "C" fn f() {}

// @has foo/fn.g.html '//div[@class="item-decl"]/pre[@class="rust"]' '#[export_name = "bar"]'
#[export_name = "bar"]
pub extern "C" fn g() {}

// @has foo/struct.Repr.html '//div[@class="item-decl"]' '#[repr(C, align(8))]'
#[repr(C, align(8))]
pub struct Repr;
