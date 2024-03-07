//@rustc-env: CLIPPY_PETS_PRINT=1
//@rustc-env: CLIPPY_PRINT_MIR=1

#[derive(Default)]
struct Example {
    owned_1: String,
    owned_2: String,
    copy_1: u32,
    copy_2: u32,
}

#[warn(clippy::borrow_pats)]
fn replace_self() {
    let mut drop = Example::default();
    drop = Example::default();

    let mut copy = 15;
    copy = 17;
}

#[warn(clippy::borrow_pats)]
fn conditional_replace_self() {
    let mut drop = Example::default();
    if true {
        drop = Example::default();
    }

    let mut copy = 15;
    if true {
        copy = 17;
    }
}

#[warn(clippy::borrow_pats)]
fn assign_copy_field() {
    let mut ex = Example::default();
    ex.copy_1 = 10;
}

#[warn(clippy::borrow_pats)]
fn assign_drop_field() {
    let mut ex = Example::default();
    ex.owned_1 = String::new();
}

#[warn(clippy::borrow_pats)]
fn move_drop_field_to_var() {
    let ex = Example::default();
    let _hey = ex.owned_1;
}

#[warn(clippy::borrow_pats)]
fn copy_field() {
    let ex = Example::default();
    let _hey = ex.copy_1;
}

#[warn(clippy::borrow_pats)]
fn move_drop_field_as_arg() {
    let ex = Example::default();
    take_string(ex.owned_1);
}

#[warn(clippy::borrow_pats)]
fn return_drop_field() -> String {
    let ex = Example::default();
    ex.owned_1
}

#[warn(clippy::borrow_pats)]
fn borrow_field_as_arg() {
    let ex = Example::default();
    take_string_ref(&ex.owned_1);
}

#[warn(clippy::borrow_pats)]
fn borrow_field_into_var() {
    let ex = Example::default();
    let _hey = &ex.owned_1;
}

fn take_string(_: String) {}
fn take_string_ref(_: &String) {}

fn main() {}
