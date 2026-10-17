//@ compile-flags: -Zon-broken-pipe=wrong
//@ check-fail

fn main() {}

//~? ERROR incorrect value `wrong` for unstable option `on-broken-pipe`
