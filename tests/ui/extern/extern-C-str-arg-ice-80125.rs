// issue: rust-lang/rust#80125
//@ check-pass
type ExternCallback = extern "C" fn(*const u8, u32, str);
//~^ WARN `extern` fn uses type `str`, which is not FFI-safe

pub struct Struct(ExternCallback);

#[no_mangle]
pub extern "C" fn register_something(bind: ExternCallback) -> Struct {
//~^ WARN `extern` fn uses type `str`, which is not FFI-safe
//~^^ WARN `extern` fn uses type `Struct`, which is not FFI-safe
    Struct(bind)
}

fn main() {}
