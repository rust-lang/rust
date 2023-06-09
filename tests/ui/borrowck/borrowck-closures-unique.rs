// Tests that a closure which requires mutable access to the referent
// of an `&mut` requires a "unique" borrow -- that is, the variable to
// be borrowed (here, `x`) will not be borrowed *mutably*, but
//  may be *immutable*, but we cannot allow
// multiple borrows.



fn get(x: &isize) -> isize {
    *x
}

fn set(x: &mut isize) -> isize {
    *x
}

fn a(x: &mut isize) {
    let c1 = || get(x);
    let c2 = || get(x);
    c1();
    c2();
}

fn b(x: &mut isize) {
    let c1 = || get(x);
    let c2 = || set(x); //~ ERROR closure requires unique access to `x`
    c1;
}

fn c(x: &mut isize) {
    let c1 = || get(x);
    let c2 = || { get(x); set(x); }; //~ ERROR closure requires unique access to `x`
    c1;
}

fn d(x: &mut isize) {
    let c1 = || set(x);
    let c2 = || set(x); //~ ERROR two closures require unique access to `x` at the same time
    c1;
}

fn e(x: &'static mut isize) {
    let c1 = |y: &'static mut isize| x = y;
    //~^ ERROR cannot assign to `x`, as it is not declared as mutable
    c1;
}

fn f(x: &'static mut isize) {
    let c1 = || x = panic!(); // OK assignment is unreachable.
    c1;
}

fn main() {
}
