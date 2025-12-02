//@ needs-rustc-debug-assertions

// https://github.com/rust-lang/rust/issues/147365
// Ensures we don't trigger debug assert by creating an empty Ident when determining whether
// the quotes are a raw lifetime.

extern "'" {} //~ ERROR invalid ABI: found `'`

extern "''" {} //~ ERROR invalid ABI: found `''`

extern "'''" {} //~ ERROR invalid ABI: found `'''`

fn main() {}
