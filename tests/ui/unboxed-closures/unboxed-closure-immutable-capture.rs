//@ check-pass
// Test that even unboxed closures that are capable of mutating their
// environment cannot mutate captured variables that have not been
// declared mutable (#18335)

fn set(x: &mut usize) { *x = 0; }

fn main() {
    let x = 0;
    move || x = 1; //~ WARNING cannot assign
    move || set(&mut x); //~ WARNING cannot borrow
    move || x = 1; //~ WARNING cannot assign
    move || set(&mut x); //~ WARNING cannot borrow
    || x = 1; //~ WARNING cannot assign
    || set(&mut x); //~ WARNING cannot borrow
    || x = 1; //~ WARNING cannot assign
    || set(&mut x); //~ WARNING cannot borrow
}
