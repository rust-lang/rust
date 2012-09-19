//! The trait for types that can be created from strings

use option::Option;

trait FromStr {
    static fn from_str(s: &str) -> Option<self>;
}

