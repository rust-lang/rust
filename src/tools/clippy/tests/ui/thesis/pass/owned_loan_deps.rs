//@rustc-env: CLIPPY_PETS_PRINT=1
//@rustc-env: CLIPPY_PRINT_MIR=1

fn extend_string(data: &String) -> &String {
    data
}

#[warn(clippy::borrow_pats)]
fn call_extend_simple() {
    let data = String::new();

    extend_string(&data).is_empty();
}

#[warn(clippy::borrow_pats)]
fn call_extend_named() {
    let data = String::new();
    let loan = &data;

    // The extention is not tracked for named loans
    extend_string(loan).is_empty();
}

fn debug() {
    let mut owned_a = String::from("=^.^=");
    let owned_b = String::from("=^.^=");
    let mut loan;

    loan = &owned_a;
    magic(loan);

    if true {
        owned_a.push('s');
    } else {
        magic(loan);
    }

    loan = &owned_b;
    magic(loan);
}

fn magic(_s: &String) {}

fn main() {}
