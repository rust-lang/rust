extern "C" {
    const fn foo();
    //~^ ERROR functions in `extern` blocks cannot have qualifiers
    const unsafe fn bar();
    //~^ ERROR functions in `extern` blocks cannot have qualifiers
    //~| ERROR items in `extern` blocks without an `unsafe` qualifier cannot have safety qualifiers
}

fn main() {}
