//! Other operating systems

use crate::detect::cache;

#[allow(dead_code)]
pub(crate) fn detect_features() -> cache::Initializer {
    cache::Initializer::default()
}
