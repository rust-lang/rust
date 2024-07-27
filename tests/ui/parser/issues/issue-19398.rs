trait T {
    extern "Rust" unsafe fn foo();
    //~^ ERROR expected `fn`, found keyword `unsafe`
    //~| NOTE expected `fn`
    //~| HELP `unsafe` must come before `extern "Rust"`
    //~| SUGGESTION unsafe extern "Rust"
    //~| NOTE keyword order for functions declaration is `pub`, `default`, `const`, `async`, `unsafe`, `extern`
}

fn main() {}
