// edition:2018

pub struct Foo;

impl Foo {
    // @has header.json "$.index[*][?(@.name=='nothing_meth')].inner.header" "[]"
    pub fn nothing_meth() {}

    // @has - "$.index[*][?(@.name=='const_meth')].inner.header" '["const"]'
    pub const fn const_meth() {}

    // @has - "$.index[*][?(@.name=='async_meth')].inner.header" '["async"]'
    pub async fn async_meth() {}

    // @count - "$.index[*][?(@.name=='async_unsafe_meth')].inner.header[*]" 2
    // @has - "$.index[*][?(@.name=='async_unsafe_meth')].inner.header[*]" '"async"'
    // @has - "$.index[*][?(@.name=='async_unsafe_meth')].inner.header[*]" '"unsafe"'
    pub async unsafe fn async_unsafe_meth() {}

    // @count - "$.index[*][?(@.name=='const_unsafe_meth')].inner.header[*]" 2
    // @has - "$.index[*][?(@.name=='const_unsafe_meth')].inner.header[*]" '"const"'
    // @has - "$.index[*][?(@.name=='const_unsafe_meth')].inner.header[*]" '"unsafe"'
    pub const unsafe fn const_unsafe_meth() {}

    // It's impossible for a method to be both const and async, so no test for that
}
