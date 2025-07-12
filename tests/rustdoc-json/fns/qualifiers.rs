//@ edition:2018

//@ jq .index[] | select(.name == "nothing_fn").inner.function.header? | [.is_async, .is_const, .is_unsafe] == [false, false, false]
pub fn nothing_fn() {}

//@ jq .index[] | select(.name == "unsafe_fn").inner.function.header? | [.is_async, .is_const, .is_unsafe] == [false, false, true]
pub unsafe fn unsafe_fn() {}

//@ jq .index[] | select(.name == "const_fn").inner.function.header? | [.is_async, .is_const, .is_unsafe] == [false, true, false]
pub const fn const_fn() {}

//@ jq .index[] | select(.name == "async_fn").inner.function.header? | [.is_async, .is_const, .is_unsafe] == [true, false, false]
pub async fn async_fn() {}

//@ jq .index[] | select(.name == "async_unsafe_fn").inner.function.header? | [.is_async, .is_const, .is_unsafe] == [true, false, true]
pub async unsafe fn async_unsafe_fn() {}

//@ jq .index[] | select(.name == "const_unsafe_fn").inner.function.header? | [.is_async, .is_const, .is_unsafe] == [false, true, true]
pub const unsafe fn const_unsafe_fn() {}

// It's impossible for a function to be both const and async, so no test for that
