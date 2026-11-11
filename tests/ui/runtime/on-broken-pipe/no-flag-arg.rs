//@ compile-flags: -Zon-broken-pipe
//@ check-fail

fn main() {}

//~? ERROR unstable option `on-broken-pipe` requires either `kill`, `error`, or `inherit`
