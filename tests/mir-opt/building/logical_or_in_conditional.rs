// skip-filecheck
//@ compile-flags: -Z validate-mir
//@ edition: 2024
struct Droppy(u8);
impl Drop for Droppy {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}

enum E {
    A(u8),
    B,
}

impl E {
    fn f() -> Self {
        Self::A(1)
    }
}

fn always_true() -> bool {
    true
}

// EMIT_MIR logical_or_in_conditional.test_or.built.after.mir
fn test_or() {
    if Droppy(0).0 > 0 || Droppy(1).0 > 1 {}
}

// EMIT_MIR logical_or_in_conditional.test_complex.built.after.mir
fn test_complex() {
    if let E::A(_) = E::f()
        && ((always_true() && Droppy(0).0 > 0) || Droppy(1).0 > 1)
    {}

    if !always_true()
        && let E::B = E::f()
    {}
}

fn main() {
    test_or();
}
