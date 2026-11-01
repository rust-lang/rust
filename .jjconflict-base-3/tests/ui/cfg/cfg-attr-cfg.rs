//@ run-pass
// main is conditionally compiled, but the conditional compilation
// is conditional too!


#[cfg_attr(FALSE, cfg(bar))]
fn main() { }
