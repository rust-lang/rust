extern "C" {
    const fn foo();
    //~^ ERROR functions in `extern` blocks cannot have qualifiers
    const unsafe fn bar();
    //~^ ERROR functions in `extern` blocks cannot have qualifiers
    //~| ERROR items in unadorned `extern` blocks cannot have safety qualifiers
}

fn main() {}
