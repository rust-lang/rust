trait T {
    extern "Rust" unsafe fn foo();
    //~^ ERROR expected `{`, found keyword `unsafe`
}

fn main() {}
