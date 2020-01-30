trait T {
    extern "Rust" unsafe fn foo(); //~ ERROR expected one of `async`, `const`
}

fn main() {}
