#[track_caller]
//~^ ERROR `#[track_caller]` can only be used with the Rust ABI
extern "C" fn c_fn() {}

#[track_caller]
extern "Rust" fn rust_fn() {}

extern "C" {
    #[track_caller]
    //~^ ERROR `#[track_caller]` can only be used with the Rust ABI
    fn c_extern();
}

extern "Rust" {
    #[track_caller]
    fn rust_extern();
}

struct S;

impl S {
    #[track_caller]
    //~^ ERROR `#[track_caller]` can only be used with the Rust ABI
    extern "C" fn c_method() {}

    #[track_caller]
    extern "Rust" fn rust_method() {}
}

#[rustfmt::skip] // rustfmt will insert the implicit "C"
#[track_caller]
//~^ ERROR `#[track_caller]` can only be used with the Rust ABI
pub extern fn extern_missing_abi() {}
//~^ WARN `extern` declarations without an explicit ABI are deprecated

fn main() {}
