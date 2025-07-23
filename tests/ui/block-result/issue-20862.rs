//@ compile-flags: -Z span_free_formats
fn foo(x: i32) {
    |y| x + y
//~^ ERROR: mismatched types
}

fn main() {
    let x = foo(5)(2);
//~^ ERROR: expected function, found `()`
}
