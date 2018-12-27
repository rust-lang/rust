// Tests that two closures cannot simultaneously have mutable
// and immutable access to the variable. Issue #6801.

fn get(x: &isize) -> isize {
    *x
}

fn set(x: &mut isize) {
    *x = 4;
}

fn a(x: &isize) {
    let mut c1 = || set(&mut *x);
    //~^ ERROR cannot borrow
    let mut c2 = || set(&mut *x);
    //~^ ERROR cannot borrow
    //~| ERROR two closures require unique access to `x` at the same time
    c2(); c1();
}

fn main() {
}
