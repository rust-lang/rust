//~ ERROR kind="static-nobundle" is unstable
// Test the behavior of rustc when non-existent library is statically linked

// compile-flags: -l static-nobundle=nonexistent

fn main() {}
