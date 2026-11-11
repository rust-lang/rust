#[deny(improper_ctypes_definitions)]

// It's an improper ctype (a slice) arg in an extern "C" fnptr.

pub type F = extern "C" fn(&[u8]);
//~^ ERROR: `extern` fn uses type `[u8]`, which is not FFI-safe


fn main() {}
