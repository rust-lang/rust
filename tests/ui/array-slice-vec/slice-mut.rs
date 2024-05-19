// Test mutability and slicing syntax.

fn main() {
    let x: &[isize] = &[1, 2, 3, 4, 5];
    // Immutable slices are not mutable.

    let y: &mut[_] = &x[2..4];
    //~^ ERROR mismatched types
    //~| expected mutable reference `&mut [_]`
    //~| found reference `&[isize]`
    //~| types differ in mutability
}
