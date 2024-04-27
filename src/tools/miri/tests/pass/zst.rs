//@compile-flags: -Zmiri-permissive-provenance
#[derive(PartialEq, Debug)]
struct A;

fn zst_ret() -> A {
    A
}

fn use_zst() -> A {
    let a = A;
    a
}

fn main() {
    // Not using the () type here, as writes of that type do not even have MIR generated.
    // Also not assigning directly as that's array initialization, not assignment.
    let zst_val = [1u8; 0];

    assert_eq!(zst_ret(), A);
    assert_eq!(use_zst(), A);
    let x = 42 as *mut [u8; 0];
    // Reading and writing is ok.
    unsafe {
        *x = zst_val;
    }
    unsafe {
        let _y = *x;
    }
}
