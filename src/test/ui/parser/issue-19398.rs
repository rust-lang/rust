trait T {
    //~^ ERROR missing `fn`, `type`, `const`, or `static` for item declaration
    extern "Rust" unsafe fn foo();
}

fn main() {}
