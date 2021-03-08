// gate-test-raw_dylib
// only-windows-msvc
#[link(name = "foo", kind = "raw-dylib")]
//~^ ERROR: kind="raw-dylib" is unstable
extern "C" {}

fn main() {}
