// only-windows
#[link(name = "foo", kind = "raw-dylib")]
//~^ ERROR: link kind `raw-dylib` is unstable
extern "C" {}

fn main() {}
