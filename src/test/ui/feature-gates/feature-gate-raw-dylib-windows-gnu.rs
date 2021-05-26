// gate-test-raw_dylib
// only-windows-gnu
#[link(name = "foo", kind = "raw-dylib")]
//~^ ERROR: kind="raw-dylib" is unstable
//~| WARNING: `#[link(...)]` with `kind = "raw-dylib"` not supported on windows-gnu
extern "C" {}

fn main() {}
