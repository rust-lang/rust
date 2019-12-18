//! Unordered containers, implemented as hash-tables

pub mod map;
pub mod set;

/// Enables deterministic hashing, which is useful when repeatability
/// is desired, e.g., during debugging. Returns `true` if no
/// `RandomState` instance was constructed prior to the first
/// invocation of `enable_deterministic_hashing`. Put another way,
/// if `enable_deterministic_hashing` returns `true`, then every
/// `HashMap` or `HashSet`'s hasher that isn't otherwise specified
/// will have been generated without using randomness.
///
/// A possible use is to structure one's program as follows:
///
/// ```
/// #![feature(deterministic_hashing)]
///
/// fn main() {
///    debug_assert!(std::collections::enable_deterministic_hashing());
///    // ...
/// }
/// ```
///
/// In this way, deterministic hashing will be enabled for debug
/// builds, but not for release builds.
///
/// Warning: `hash_builder` is normally randomly generated, and is
/// designed to allow HashMaps to be resistant to attacks that cause
/// many collisions and very poor performance. Using this function
/// can expose a DoS attack vector.
#[unstable(feature = "deterministic_hashing", reason = "new API", issue = "0")]
pub fn enable_deterministic_hashing() -> bool {
    use crate::sync::atomic::Ordering;
    let flags = map::HASHING_FLAGS.fetch_or(map::DETERMINISTIC_HASHING_ENABLED, Ordering::SeqCst);
    (flags & map::RANDOM_STATE_CONSTRUCTED_BEFORE_DETERMINISTIC_HASHING_ENABLED) == 0
}
