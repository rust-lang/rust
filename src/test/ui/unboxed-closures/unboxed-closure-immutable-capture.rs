// Test that even unboxed closures that are capable of mutating their
// environment cannot mutate captured variables that have not been
// declared mutable (#18335)

fn set(x: &mut usize) { *x = 0; }

fn main() {
    let x = 0;
    move || x = 1; //~ ERROR cannot assign
    move || set(&mut x); //~ ERROR cannot borrow
    move || x = 1; //~ ERROR cannot assign
    move || set(&mut x); //~ ERROR cannot borrow
    || x = 1; //~ ERROR cannot assign
    // FIXME: this should be `cannot borrow` (issue #18330)
    || set(&mut x); //~ ERROR cannot assign
    || x = 1; //~ ERROR cannot assign
    // FIXME: this should be `cannot borrow` (issue #18330)
    || set(&mut x); //~ ERROR cannot assign
}
