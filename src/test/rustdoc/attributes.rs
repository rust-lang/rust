#![crate_name = "foo"]

// @has foo/fn.f.html '//*[@class="docblock attributes"]' '#[no_mangle]'
#[no_mangle]
pub extern "C" fn f() {}

// @has foo/fn.g.html '//*[@class="docblock attributes"]' '#[export_name = "bar"]'
#[export_name = "bar"]
pub extern "C" fn g() {}

// @has foo/enum.Foo.html '//*[@class="docblock attributes top-attr"]' '#[repr(i64)]'
// @has foo/enum.Foo.html '//*[@class="docblock attributes top-attr"]' '#[must_use]'
#[repr(i64)]
#[must_use]
pub enum Foo {
    Bar,
}
