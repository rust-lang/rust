//@ compile-flags: -Zon-broken-pipe=default
//@ check-fail

fn main() {}

//~? ERROR incorrect value `default` for unstable option `on-broken-pipe`
