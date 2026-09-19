//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

// https://github.com/rust-lang/rust/issues/19181

// rustdoc should not panic when target crate has compilation errors

fn main() { 0 }
