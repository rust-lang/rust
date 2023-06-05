// ignore-windows
// compile-flags: --crate-type lib
#[link(name = "foo", kind = "raw-dylib")]
//~^ ERROR: link kind `raw-dylib` is only supported on Windows targets
extern "C" {}
