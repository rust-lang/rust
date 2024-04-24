extern "C" unsafe {
    //~^ ERROR expected `{`, found keyword `unsafe`
    unsafe fn foo();
    //~^ ERROR functions in `extern` blocks cannot have qualifiers
}

fn main() {}
