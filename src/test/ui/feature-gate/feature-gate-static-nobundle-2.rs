//~ ERROR kind="static-nobundle" is feature gated
// Test the behavior of rustc when non-existent library is statically linked

// compile-flags: -l static-nobundle=nonexistent

fn main() {}
