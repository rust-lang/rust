// issue: rust-lang/rust#80125
//@ check-pass
type ExternCallback = extern "C" fn(*const u8, u32, str);
//~^ WARN `extern` callback uses type `str`, which is not FFI-safe

pub struct Struct(ExternCallback);

#[no_mangle]
pub extern "C" fn register_something(bind: ExternCallback) -> Struct {
// ^ FIXME: the error isn't seen here, but at least it's reported elsewhere
//~^^ WARN `extern` fn uses type `Struct`, which is not FFI-safe
    Struct(bind)
}

fn main() {}
