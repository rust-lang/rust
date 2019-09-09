//! Other operating systems

use crate::detect::cache;

pub(crate) fn detect_features() -> cache::Initializer {
    cache::Initializer::default()
}
