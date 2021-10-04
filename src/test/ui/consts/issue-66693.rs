// Tests that the compiler does not ICE when const-evaluating a `panic!()` invocation with a
// non-`&str` argument.

const _: () = panic!(1);
//~^ ERROR: argument to `panic!()` in a const context must have type `&str`

static _FOO: () = panic!(true);
//~^ ERROR: argument to `panic!()` in a const context must have type `&str`

const fn _foo() {
    panic!(&1); //~ ERROR: argument to `panic!()` in a const context must have type `&str`
}

// ensure that conforming panics don't cause an error
const _: () = panic!();
static _BAR: () = panic!("panic in static");

const fn _bar() {
    panic!("panic in const fn");
}

fn main() {}
