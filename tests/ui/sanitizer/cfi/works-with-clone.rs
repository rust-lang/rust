// Verifies that types with builtin Clone implementations (i.e., arrays, tuples,
// and closures) can be cloned.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

fn main() {
    // The CloneShims for the values below are not transformed, as the Clone trait is not dyn
    // compatible.
    let array = [1i32, 2, 3];
    assert_eq!(array.clone(), array);
    let tuple = (1i32, 2u8);
    assert_eq!(tuple.clone(), tuple);
    let x = 1i32;
    let closure = move || x;
    assert_eq!(closure.clone()(), closure());
}
