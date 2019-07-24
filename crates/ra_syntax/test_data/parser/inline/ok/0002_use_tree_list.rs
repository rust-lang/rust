use {crate::path::from::root, or::path::from::crate_name}; // Rust 2018 (with a crate named `or`)
use {path::from::root}; // Rust 2015
use ::{some::arbritrary::path}; // Rust 2015
use ::{{{crate::export}}}; // Nonsensical but perfectly legal nestnig
