fn main() {
    super let a = 1;
    //~^ ERROR `super let` is experimental
}

// Check that it also isn't accepted in cfg'd out code.
#[cfg(any())]
fn a() {
    super let a = 1;
    //~^ ERROR `super let` is experimental
}
