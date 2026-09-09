#![crate_name = "dep"]
#![crate_type = "rlib"]

// The private code that gets appended to between compilations lives in this
// submodule's own file (`sub.rs`), *not* in this crate-root file. The crate
// root's span covers the whole of `dep.rs`, so any byte appended here would be
// hashed unconditionally; appending to a submodule file is what actually
// exercises "private code added at the end of a file".
pub mod sub;

pub fn public() {}
