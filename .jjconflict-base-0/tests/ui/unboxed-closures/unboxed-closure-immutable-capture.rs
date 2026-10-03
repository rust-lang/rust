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
    || set(&mut x); //~ ERROR cannot borrow
    || x = 1; //~ ERROR cannot assign
    || set(&mut x); //~ ERROR cannot borrow
}
