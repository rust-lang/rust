// Test that an unboxed closure that mutates a free variable will
// cause borrow conflicts.



fn main() {
    let mut x = 0;
    let f = || x += 1;
    let _y = x; //~ ERROR cannot use `x` because it was mutably borrowed
    f;
}
