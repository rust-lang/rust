//@ only-aarch64
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

///```ignore-x86_64
/// assert!(cfg!(not(target_arch = "x86_64")));
///```
pub fn foo() -> u8 {
    4
}

fn main() {}
