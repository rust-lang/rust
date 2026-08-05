//! Test that name resolution works correctly when a struct and its constructor
//! function have the same name within a nested scope. This checks that the
//! compiler can distinguish between type names and value names in the same
//! namespace.

//@ run-pass

struct Point {
    i: isize,
}

impl Point {
    fn get_value(&self) -> isize {
        return 37;
    }
}

// Constructor function with the same name as the struct
#[allow(non_snake_case)]
fn Point(i: isize) -> Point {
    Point { i }
}

pub fn main() {
    // Test that we can use the constructor function
    let point = Point(42);
    assert_eq!(point.i, 42);
    assert_eq!(point.get_value(), 37);
}
