extern "C" unsafe {
    //~^ ERROR expected `{`, found keyword `unsafe`
    //~| ERROR extern block cannot be declared unsafe
    unsafe fn foo();
    //~^ ERROR items in unadorned `extern` blocks cannot have safety qualifiers
}

fn main() {}
