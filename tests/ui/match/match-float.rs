//@ run-pass
//@ compile-flags: --check-cfg=cfg(target_has_reliable_f16,target_has_reliable_f128)
// Makes sure we use `==` (not bitwise) semantics for float comparison.

#![feature(cfg_target_has_reliable_f16_f128)]
#![feature(f128)]
#![feature(f16)]

#[cfg(target_has_reliable_f16)]
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

#[cfg(target_has_reliable_f128)]
fn check_f128() {
    const F1: f128 = 0.0;
    const F2: f128 = -0.0;
    assert_eq!(F1, F2);
    assert_ne!(F1.to_bits(), F2.to_bits());
    assert!(matches!(F1, F2));
    assert!(matches!(F2, F1));
}

fn main() {
    #[cfg(target_has_reliable_f16)]
    check_f16();
    check_f32();
    check_f64();
    #[cfg(target_has_reliable_f128)]
    check_f128();
}
