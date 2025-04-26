//@ check-pass
//@ compile-flags: -Z unpretty=expanded
//@ edition: 2015

#![feature(core_intrinsics, generic_assert)]

fn arbitrary_consuming_method_for_demonstration_purposes() {
    let elem = 1i32;
    assert!(elem as usize);
}

fn addr_of() {
    let elem = 1i32;
    assert!(&elem);
}

fn binary() {
    let elem = 1i32;
    assert!(elem == 1);
    assert!(elem >= 1);
    assert!(elem > 0);
    assert!(elem < 3);
    assert!(elem <= 3);
    assert!(elem != 3);
}

fn unary() {
    let elem = &1i32;
    assert!(*elem);
}

fn main() {
}
