// Test that we give a note when the old LUB/GLB algorithm would have
// succeeded but the new code (which is stricter) gives an error.

fn foo(
    x: fn(&u8, &u8),
    y: for<'a> fn(&'a u8, &'a u8),
) {
    let z = match 22 { //~ ERROR incompatible types
        0 => x,
        _ => y,
    };
}

fn bar(
    x: fn(&u8, &u8),
    y: for<'a> fn(&'a u8, &'a u8),
) {
    let z = match 22 {
        // No error with an explicit cast:
        0 => x as for<'a> fn(&'a u8, &'a u8),
        _ => y,
    };
}

fn main() {
}
