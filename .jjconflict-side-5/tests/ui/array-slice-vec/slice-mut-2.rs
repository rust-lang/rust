// Test mutability and slicing syntax.

fn main() {
    let x: &[isize] = &[1, 2, 3, 4, 5];
    // Can't mutably slice an immutable slice
    let slice: &mut [isize] = &mut [0, 1];
    let _ = &mut x[2..4]; //~ERROR cannot borrow `*x` as mutable, as it is behind a `&` reference
}
