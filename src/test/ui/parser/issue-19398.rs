trait T {
    //~^ ERROR missing `fn`, `type`, or `const` for associated-item declaration
    extern "Rust" unsafe fn foo();
}

fn main() {}
