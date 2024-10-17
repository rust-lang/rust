mod num_traits;

pub use num_traits::{Float, Hex, Int};

// List of all files present in libm's source
include!(concat!(env!("OUT_DIR"), "/all_files.rs"));
