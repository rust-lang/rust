trait T {
    extern "Rust" unsafe fn foo(); //~ ERROR expected `fn`, found keyword `unsafe`
}

fn main() {}
