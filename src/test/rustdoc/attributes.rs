#![crate_name = "foo"]

// @has foo/fn.f.html '//*[@class="docblock attributes"]' '#[no_mangle]'
#[no_mangle]
pub extern "C" fn f() {}

// @has foo/fn.g.html '//*[@class="docblock attributes"]' '#[export_name = "bar"]'
#[export_name = "bar"]
pub extern "C" fn g() {}

// @matches foo/enum.Foo.html '//*[@class="docblock attributes top-attr"]' \
//      '(?m)\A#\[repr\(i64\)\]\n#\[must_use\]\Z'
#[repr(i64)]
#[must_use]
pub enum Foo {
    Bar,
}

// @has foo/struct.Repr.html '//*[@class="docblock attributes top-attr"]' '#[repr(C, align(8))]'
#[repr(C, align(8))]
pub struct Repr;
