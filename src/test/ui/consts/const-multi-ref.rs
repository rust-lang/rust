// Ensure that we point the user to the erroneous borrow but not to any subsequent borrows of that
// initial one.

const _: i32 = {
    let mut a = 5;
    let p = &mut a; //~ ERROR mutable references are not allowed in constants

    let reborrow = {p};
    let pp = &reborrow;
    let ppp = &pp;
    ***ppp
};

const _: std::cell::Cell<i32> = {
    let mut a = std::cell::Cell::new(5);
    let p = &a; //~ ERROR borrowed element may contain interior mutability

    let reborrow = {p};
    let pp = &reborrow;
    let ppp = &pp;
    a
};

fn main() {}
