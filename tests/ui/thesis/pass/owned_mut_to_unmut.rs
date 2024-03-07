//@rustc-env: CLIPPY_PETS_PRINT=1
//@rustc-env: CLIPPY_PRINT_MIR=1

#[warn(clippy::borrow_pats)]
fn mut_move_to_immut() {
    let mut x = "Hello World".to_string();
    x.push('x');
    x.clear();
    let x2 = x;
    let _ = x2.len();
}

#[warn(clippy::borrow_pats)]
fn init_with_mut_block() {
    let x2 = {
        let mut x = "Hello World".to_string();
        x.push('x');
        x.clear();
        x
    };
    let _ = x2.len();
}

#[warn(clippy::borrow_pats)]
fn mut_copy_to_immut_shadow() {
    let mut counter = 1;
    counter += 10;

    let counter = counter;

    let _ = counter;
}

#[warn(clippy::borrow_pats)]
fn mut_copy_to_immut() {
    let mut counter = 1;
    counter += 10;

    let snapshot = counter;

    let _ = snapshot;
}

#[warn(clippy::borrow_pats)]
fn mut_copy_to_immut_and_use_after() {
    let mut counter = 1;
    counter += 10;

    let snapshot = counter;

    counter += 3;

    let _ = snapshot + counter;
}

#[warn(clippy::borrow_pats)]
fn main() {
    let mut s = String::new();
    s += "Hey";

    let _ = s;
}
