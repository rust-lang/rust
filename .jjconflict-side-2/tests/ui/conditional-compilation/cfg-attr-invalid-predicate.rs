#[cfg(foo(bar))] //~ ERROR malformed `cfg` attribute input [E0539]
fn check() {}

fn main() {}
