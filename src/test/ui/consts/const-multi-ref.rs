// Ensure that we point the user to the erroneous borrow but not to any subsequent borrows of that
// initial one.

const _X: i32 = {
    let mut a = 5;
    let p = &mut a; //~ ERROR references in constants may only refer to immutable values

    let reborrow = {p};
    let pp = &reborrow;
    let ppp = &pp;
    ***ppp
};

fn main() {}
