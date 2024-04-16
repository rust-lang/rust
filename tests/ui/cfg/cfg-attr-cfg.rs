//@ run-pass
// main is conditionally compiled, but the conditional compilation
// is conditional too!

//@ pretty-expanded FIXME #23616

#[cfg_attr(FALSE, cfg(bar))]
fn main() { }
