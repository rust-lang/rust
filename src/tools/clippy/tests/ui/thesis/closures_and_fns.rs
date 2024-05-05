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
fn use_local_func() {
    let func = take_string_ref;

    func(&String::new());
}

#[warn(clippy::borrow_pats)]
fn use_arg_func(func: fn(&String) -> &str) {
    func(&String::new());
}

#[warn(clippy::borrow_pats)]
fn call_closure_with_arg(s: String) {
    let close = |s: &String| s.len();
    close(&s);
}

#[warn(clippy::borrow_pats)]
fn call_closure_borrow_env(s: String) {
    let close = || s.len();
    close();
}

#[warn(clippy::borrow_pats)]
fn call_closure_move_s(s: String) {
    let close = move || s.len();
    close();
}

#[warn(clippy::borrow_pats)]
fn call_closure_move_field(ex: Example) {
    let close = move || ex.owned_1.len();
    close();
}

fn take_string(_s: String) {}
fn take_string_ref(_s: &String) {}
fn pass_t<T>(tee: T) -> T {
    tee
}

fn main() {}
