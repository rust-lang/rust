//! The trait for types that can be created from strings

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use option::Option;

pub trait FromStr {
    static pure fn from_str(s: &str) -> Option<self>;
}

