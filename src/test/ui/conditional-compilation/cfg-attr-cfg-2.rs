//
// error-pattern: `main` function not found
// compile-flags: --cfg foo

// main is conditionally compiled, but the conditional compilation
// is conditional too!

#[cfg_attr(foo, cfg(bar))]
fn main() { }
