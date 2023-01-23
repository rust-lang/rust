//! Horizon-specific extensions for working with the [`std::thread`] module.
//!
//! [`std::thread`]: crate::thread

#![unstable(feature = "thread_scheduling", issue = "none")]

use crate::fmt;

/// The relative priority of the thread. See the `libctru` docs for the `prio`
/// parameter of [`threadCreate`] for details on valid values.
///
/// [`threadCreate`]: https://libctru.devkitpro.org/thread_8h.html#a38c873d8cb02de7f5eca848fe68183ee
pub struct Priority(pub(crate) libc::c_int);

impl TryFrom<i32> for Priority {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0x18..=0x3F => Ok(Self(value)),
            _ => Err(()),
        }
    }
}

impl crate::fmt::Debug for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Priority").finish_non_exhaustive()
    }
}

/// The CPU(s) on which to spawn the thread. See the `libctru` docs for the
/// `core_id` parameter of [`threadCreate`] for details on valid values.
///
/// [`threadCreate`]: https://libctru.devkitpro.org/thread_8h.html#a38c873d8cb02de7f5eca848fe68183ee
pub struct Affinity(pub(crate) libc::c_int);

impl Default for Affinity {
    fn default() -> Self {
        Self(-2)
    }
}

impl TryFrom<i32> for Affinity {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            -2..=4 => Ok(Self(value)),
            _ => Err(()),
        }
    }
}

impl crate::fmt::Debug for Affinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Affinity").finish_non_exhaustive()
    }
}
