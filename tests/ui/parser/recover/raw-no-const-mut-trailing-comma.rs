// Regression test for https://github.com/rust-lang/rust/issues/157950.

fn main() {
    takes_raw_ptr_args(&raw x,)
    //~^ ERROR expected one of
    //~| ERROR cannot find function `takes_raw_ptr_args` in this scope
}
