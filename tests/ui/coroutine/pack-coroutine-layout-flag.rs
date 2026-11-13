//@ edition: 2021
//@ revisions: classic captures_only invalid
//@ [classic] check-pass
//@ [classic] compile-flags: -Z pack-coroutine-layout=classic
//@ [captures_only] check-pass
//@ [captures_only] compile-flags: -Z pack-coroutine-layout=captures-only
//@ [invalid] compile-flags: -Z pack-coroutine-layout=invalid
//[invalid]~? ERROR incorrect value `invalid` for unstable option `pack-coroutine-layout` - either `classic` or `captures-only` was expected

#![crate_type = "lib"]

pub async fn example(x: String) {
    drop(x);
}
