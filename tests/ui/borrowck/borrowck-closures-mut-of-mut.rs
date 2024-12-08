// Tests that two closures cannot simultaneously both have mutable
// access to the variable. Related to issue #6801.

fn get(x: &isize) -> isize {
    *x
}

fn set(x: &mut isize) {
    *x = 4;
}

fn a(x: &mut isize) {
    let mut c1 = || set(&mut *x);
    let mut c2 = || set(&mut *x);
    //~^ ERROR two closures require unique access to `x` at the same time
    c2(); c1();
}

fn main() {
}
