extern "C" unsafe {
               //~^ ERROR expected `{`, found keyword `unsafe`
               //~| ERROR extern block cannot be declared unsafe
    unsafe fn foo();
        //~^ ERROR functions in `extern` blocks cannot have qualifiers
}

fn main() {}
