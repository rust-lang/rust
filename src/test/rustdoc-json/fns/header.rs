// edition:2018

// @has header.json "$.index[*][?(@.name=='nothing_fn')].inner.header" "[]"
pub fn nothing_fn() {}

// @has - "$.index[*][?(@.name=='const_fn')].inner.header" '["const"]'
pub const fn const_fn() {}

// @has - "$.index[*][?(@.name=='async_fn')].inner.header" '["async"]'
pub async fn async_fn() {}

// @count - "$.index[*][?(@.name=='async_unsafe_fn')].inner.header[*]" 2
// @has - "$.index[*][?(@.name=='async_unsafe_fn')].inner.header[*]" '"async"'
// @has - "$.index[*][?(@.name=='async_unsafe_fn')].inner.header[*]" '"unsafe"'
pub async unsafe fn async_unsafe_fn() {}

// @count - "$.index[*][?(@.name=='const_unsafe_fn')].inner.header[*]" 2
// @has - "$.index[*][?(@.name=='const_unsafe_fn')].inner.header[*]" '"const"'
// @has - "$.index[*][?(@.name=='const_unsafe_fn')].inner.header[*]" '"unsafe"'
pub const unsafe fn const_unsafe_fn() {}

// It's impossible for a function to be both const and async, so no test for that
