// Tests that the compiler does not ICE when const-evaluating a `panic!()` invocation with a
// non-`&str` argument.

const _: () = panic!(1);
//~^ ERROR: argument to `panic!()` in a const context must have type `&str`

static _FOO: () = panic!(true);
//~^ ERROR: argument to `panic!()` in a const context must have type `&str`

const fn _foo() {
    panic!(&1);
    //~^ ERROR: argument to `panic!()` in a const context must have type `&str`
    //~| ERROR: erroneous constant used [const_err]
    //~| WARNING: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

// ensure that conforming panics don't cause an error
const _: () = panic!();
static _BAR: () = panic!("panic in static");

const fn _bar() {
    panic!("panic in const fn");
}

fn main() {}
