fn foo(x: i32) {
    |y| x + y
//~^ ERROR: mismatched types
}

fn main() {
    let x = foo(5)(2);
//~^ ERROR: expected function, found `()`
}
