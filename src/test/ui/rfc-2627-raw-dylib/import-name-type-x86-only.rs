// only-windows
// ignore-x86
#[link(name = "foo", kind = "raw-dylib", import_name_type = "decorated")]
//~^ ERROR import name type is only supported on x86
extern "C" { }

fn main() {}
