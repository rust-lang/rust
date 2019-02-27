// Test that we give a note when the old LUB/GLB algorithm would have
// succeeded but the new code (which requires equality) gives an
// error. However, now that we handle subtyping correctly, we no
// longer get an error, because we recognize these two types as
// equivalent!
//
// Whoops -- now that we reinstituted the leak-check, we get an error
// again.

fn foo(
    x: fn(&u8, &u8),
    y: for<'a> fn(&'a u8, &'a u8),
) {
    let z = match 22 {
        0 => x,
        _ => y, //~ ERROR match arms have incompatible types
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
