extern "C" {
    const fn foo();
    //~^ ERROR functions in `extern` blocks cannot
    const unsafe fn bar();
    //~^ ERROR functions in `extern` blocks cannot
    //~| ERROR items in `extern` blocks without an `unsafe` qualifier cannot
}

fn main() {}
