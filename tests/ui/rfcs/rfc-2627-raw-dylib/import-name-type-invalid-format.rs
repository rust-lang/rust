// only-windows
// only-x86
#[link(name = "foo", kind = "raw-dylib", import_name_type = 6)]
//~^ ERROR import name type must be of the form `import_name_type = "string"`
extern "C" { }

fn main() {}
