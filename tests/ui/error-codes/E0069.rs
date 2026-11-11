fn foo() -> u8 {
    return;
    //~^ ERROR `return;` in a function whose return type is not `()`
}

fn main() {
}
