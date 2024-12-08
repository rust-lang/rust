//@ edition:2021
//@ run-pass

// Test that we can mutate a place through a mut-borrow
// that is captured by the closure

// Check that we can mutate when one deref is required
fn mut_ref_1() {
    let mut x = String::new();
    let rx = &mut x;

    let mut c = || {
        *rx = String::new();
    };

    c();
}

// Similar example as mut_ref_1, we don't deref the imm-borrow here,
// and so we are allowed to mutate.
fn mut_ref_2() {
    let x = String::new();
    let y = String::new();
    let mut ref_x = &x;
    let m_ref_x = &mut ref_x;

    let mut c = || {
        *m_ref_x = &y;
    };

    c();
}

// Check that we can mutate when multiple derefs of mut-borrows are required to reach
// the target place.
// It works because all derefs are mutable, if either of them was an immutable
// borrow, then we would not be able to deref.
fn mut_mut_ref() {
    let mut x = String::new();
    let mut mref_x = &mut x;
    let m_mref_x = &mut mref_x;

    let mut c = || {
        **m_mref_x = String::new();
    };

    c();
}

fn main() {
    mut_ref_1();
    mut_ref_2();
    mut_mut_ref();
}
