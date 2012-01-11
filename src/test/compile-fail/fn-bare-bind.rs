fn f() {
}

fn main() {
    // Can't produce a bare function by binding
    let g: native fn() = bind f();
    //!^ ERROR mismatched types: expected `native fn()` but found `fn@()`
}
