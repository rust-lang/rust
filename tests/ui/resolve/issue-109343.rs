// Verify that we do not ICE when failing to resolve a doc-link.
#![crate_type = "lib"]

extern crate f;
//~^ ERROR E0463

pub use inner::f;
//~^ ERROR E0432

/// [mod@std::env] [g]
pub use f as g;
//~^ ERROR pub_use_of_private_extern_crate
//~| WARN this was previously accepted by the compiler
