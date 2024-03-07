//@rustc-env: CLIPPY_PETS_PRINT=1
//@rustc-env: CLIPPY_STATS_PRINT=1
//@rustc-env: CLIPPY_PRINT_MIR=1

#![warn(clippy::borrow_pats)]

fn temp_borrow_arg_1(owned: String) {
    owned.len();
    owned.is_empty();
}

fn temp_borrow_mut_arg_1(mut owned: String) {
    owned.pop();
    owned.push('x');
    owned.clear();
}

fn temp_borrow_mixed_arg_1(mut owned: String) {
    owned.is_empty();
    owned.clear();
    owned.len();
}

fn temp_borrow_local_1() {
    let owned = String::new();
    owned.len();
    owned.is_empty();
}

fn temp_borrow_mut_local_1() {
    let mut owned = String::new();
    owned.pop();
    owned.push('x');
    owned.clear();
}

fn temp_borrow_mixed_local_1() {
    let mut owned = String::new();
    owned.is_empty();
    owned.clear();
    owned.len();
}

fn temp_borrow_mixed_mutltiple_1(a: String, mut b: String) {
    let c = String::new();
    let mut d = String::new();

    // TempBorrow
    a.is_empty();
    b.is_empty();
    c.is_empty();
    d.is_empty();

    // TempBorrowMut
    b.clear();
    d.clear();

    // TempBorrow
    a.is_empty();
    b.is_empty();
    c.is_empty();
    d.is_empty();
}

fn main() {}
