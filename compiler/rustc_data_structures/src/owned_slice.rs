use std::{borrow::Borrow, ops::Deref};

// Use our fake Send/Sync traits when on not parallel compiler,
// so that `OwnedSlice` only implements/requires Send/Sync
// for parallel compiler builds.
use crate::sync::{Send, Sync};

/// An owned slice.
///
/// This is similar to `Box<[u8]>` but allows slicing and using anything as the
/// backing buffer.
///
/// See [`slice_owned`] for `OwnedSlice` construction and examples.
///
/// ---------------------------------------------------------------------------
///
/// This is essentially a replacement for `owning_ref` which is a lot simpler
/// and even sound! 🌸
pub struct OwnedSlice {
    /// This is conceptually a `&'self.owner [u8]`.
    bytes: *const [u8],

    // +---------------------------------------+
    // | We expect `dead_code` lint here,      |
    // | because we don't want to accidentally |
    // | touch the owner — otherwise the owner |
    // | could invalidate out `bytes` pointer  |
    // |                                       |
    // | so be quiet                           |
    // +----+  +-------------------------------+
    //       \/
    //      ⊂(´･◡･⊂ )∘˚˳° (I am the phantom remnant of #97770)
    #[expect(dead_code)]
    owner: Box<dyn Send + Sync>,
}

/// Makes an [`OwnedSlice`] out of an `owner` and a `slicer` function.
///
/// ## Examples
///
/// ```rust
/// # use rustc_data_structures::owned_slice::{OwnedSlice, slice_owned};
/// let vec = vec![1, 2, 3, 4];
///
/// // Identical to slicing via `&v[1..3]` but produces an owned slice
/// let slice: OwnedSlice = slice_owned(vec, |v| &v[1..3]);
/// assert_eq!(&*slice, [2, 3]);
/// ```
///
/// ```rust
/// # use rustc_data_structures::owned_slice::{OwnedSlice, slice_owned};
/// # use std::ops::Deref;
/// let vec = vec![1, 2, 3, 4];
///
/// // Identical to slicing via `&v[..]` but produces an owned slice
/// let slice: OwnedSlice = slice_owned(vec, Deref::deref);
/// assert_eq!(&*slice, [1, 2, 3, 4]);
/// ```
pub fn slice_owned<O, F>(owner: O, slicer: F) -> OwnedSlice
where
    O: Send + Sync + 'static,
    F: FnOnce(&O) -> &[u8],
{
    try_slice_owned(owner, |x| Ok::<_, !>(slicer(x))).into_ok()
}

/// Makes an [`OwnedSlice`] out of an `owner` and a `slicer` function that can fail.
///
/// See [`slice_owned`] for the infallible version.
pub fn try_slice_owned<O, F, E>(owner: O, slicer: F) -> Result<OwnedSlice, E>
where
    O: Send + Sync + 'static,
    F: FnOnce(&O) -> Result<&[u8], E>,
{
    // We box the owner of the bytes, so it doesn't move.
    //
    // Since the owner does not move and we don't access it in any way
    // before drop, there is nothing that can invalidate the bytes pointer.
    //
    // Thus, "extending" the lifetime of the reference returned from `F` is fine.
    // We pretend that we pass it a reference that lives as long as the returned slice.
    //
    // N.B. the HRTB on the `slicer` is important — without it the caller could provide
    // a short lived slice, unrelated to the owner.

    let owner = Box::new(owner);
    let bytes = slicer(&*owner)?;

    Ok(OwnedSlice { bytes, owner })
}

impl Deref for OwnedSlice {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        // Safety:
        // `self.bytes` is valid per the construction in `slice_owned`
        // (which is the only constructor)
        unsafe { &*self.bytes }
    }
}

impl Borrow<[u8]> for OwnedSlice {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self
    }
}

// Safety: `OwnedSlice` is conceptually `(&'self.1 [u8], Box<dyn Send + Sync>)`, which is `Send`
#[cfg(parallel_compiler)]
unsafe impl Send for OwnedSlice {}

// Safety: `OwnedSlice` is conceptually `(&'self.1 [u8], Box<dyn Send + Sync>)`, which is `Sync`
#[cfg(parallel_compiler)]
unsafe impl Sync for OwnedSlice {}

#[cfg(test)]
mod tests;
