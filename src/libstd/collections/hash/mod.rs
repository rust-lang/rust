//! Unordered containers, implemented as hash-tables

pub mod map;
pub mod set;

/// Returns `true` if deterministic hashing was successfully
/// enabled. A call to `enable_deterministic_hashing` will fail (i.e.,
/// return `false`) if a `RandomState` instance was already constructed
/// while deterministic hashing was disabled.
///
/// Deterministic hashing is useful when repeatability is desired,
/// e.g., during debugging. A possible use is to structure one's
/// program as follows:
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
    let flags = map::HASHING_FLAGS.compare_and_swap(
        0,
        map::DETERMINISTIC_HASHING_ENABLED,
        Ordering::SeqCst,
    );
    flags == 0 || (flags & map::DETERMINISTIC_HASHING_ENABLED) != 0
}
