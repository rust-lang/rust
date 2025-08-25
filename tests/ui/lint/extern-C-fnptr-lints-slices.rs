#[deny(improper_c_callbacks)]

// It's an improper ctype (a slice) arg in an extern "C" fnptr.

pub type F = extern "C" fn(&[u8]);
//~^ ERROR: `extern` callback uses type `[u8]`, which is not FFI-safe


fn main() {}
