// only-windows
// only-x86
#[link(name = "foo", kind = "raw-dylib", import_name_type = "unknown")]
//~^ ERROR unknown import name type `unknown`, expected one of: decorated, noprefix, undecorated
extern "C" { }

fn main() {}
