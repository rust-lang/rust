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
    unsafe { *x = zst_val; }
    unsafe { let _y = *x; }

    // We should even be able to use "true" pointers for ZST when the allocation has been
    // removed already. The box is for a non-ZST to make sure there actually is an allocation.
    let mut x_box = Box::new(((), 1u8));
    let x = &mut x_box.0 as *mut _ as *mut [u8; 0];
    drop(x_box);
    // Reading and writing is ok.
    unsafe { *x = zst_val; }
    unsafe { let _y = *x; }
}
