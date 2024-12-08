//@ edition:2021

// Test that we can't mutate a place if we need to deref an imm-borrow
// to reach it.

fn imm_mut_ref() {
    let mut x = String::new();
    let y = String::new();
    let mref_x = &mut x;
    let ref_mref_x = &mref_x;

    let c = || {
    //~^ ERROR: cannot borrow `**ref_mref_x` as mutable, as it is behind a `&` reference
        **ref_mref_x = y;
    };

    c();
}

fn mut_imm_ref() {
    let x = String::new();
    let y = String::new();
    let mut ref_x = &x;
    let mref_ref_x = &mut ref_x;

    let c = || {
    //~^ ERROR: cannot borrow `**mref_ref_x` as mutable, as it is behind a `&` reference
        **mref_ref_x = y;
    };

    c();
}

fn main() {
    imm_mut_ref();
    mut_imm_ref();
}
