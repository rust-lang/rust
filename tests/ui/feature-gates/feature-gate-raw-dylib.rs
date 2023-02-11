// only-windows
// only-x86
#[link(name = "foo", kind = "raw-dylib")]
//~^ ERROR: link kind `raw-dylib` is unstable on x86
extern "C" {}

fn main() {}
