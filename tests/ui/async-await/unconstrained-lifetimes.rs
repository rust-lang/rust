//@ edition: 2021

// Make sure we don't complain about the implicit `-> impl Future` capturing an
// unconstrained anonymous lifetime.

async fn foo(_: Missing<'_>) {}
//~^ ERROR cannot find type `Missing` in this scope

fn main() {}
