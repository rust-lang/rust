//@ run-pass
// Makes sure we use `==` (not bitwise) semantics for float comparison.

#![feature(f128)]
#![feature(f16)]

// FIXME(f16_f128): remove gates when ABI issues are resolved

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn check_f16() {
    const F1: f16 = 0.0;
    const F2: f16 = -0.0;
    assert_eq!(F1, F2);
    assert_ne!(F1.to_bits(), F2.to_bits());
    assert!(matches!(F1, F2));
    assert!(matches!(F2, F1));
}

fn check_f32() {
    const F1: f32 = 0.0;
    const F2: f32 = -0.0;
    assert_eq!(F1, F2);
    assert_ne!(F1.to_bits(), F2.to_bits());
    assert!(matches!(F1, F2));
    assert!(matches!(F2, F1));
}

fn check_f64() {
    const F1: f64 = 0.0;
    const F2: f64 = -0.0;
    assert_eq!(F1, F2);
    assert_ne!(F1.to_bits(), F2.to_bits());
    assert!(matches!(F1, F2));
    assert!(matches!(F2, F1));
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn check_f128() {
    const F1: f128 = 0.0;
    const F2: f128 = -0.0;
    assert_eq!(F1, F2);
    assert_ne!(F1.to_bits(), F2.to_bits());
    assert!(matches!(F1, F2));
    assert!(matches!(F2, F1));
}

fn main() {
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    check_f16();
    check_f32();
    check_f64();
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    check_f128();
}
