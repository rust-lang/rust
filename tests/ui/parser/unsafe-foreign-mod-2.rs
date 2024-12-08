extern "C" unsafe {
    //~^ ERROR expected `{`, found keyword `unsafe`
    unsafe fn foo();
}

fn main() {}
