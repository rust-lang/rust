#[cfg(not(sanitize = "thread"))]
//~^ ERROR `cfg(sanitize)` is experimental
fn main() {}
