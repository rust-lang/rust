extern "C" {
    const fn foo();
    //~^ ERROR functions in `extern` blocks cannot have qualifiers
    const unsafe fn bar();
    //~^ ERROR functions in `extern` blocks cannot have qualifiers
}

fn main() {}
