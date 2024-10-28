use std::fmt;
use std::num::NonZeroU32;

/// The "raw-id" is used for interned keys in salsa -- it is basically
/// a newtype'd u32. Typically, it is wrapped in a type of your own
/// devising. For more information about interned keys, see [the
/// interned key RFC][rfc].
///
/// # Creating a `InternId`
//
/// InternId values can be constructed using the `From` impls,
/// which are implemented for `u32` and `usize`:
///
/// ```
/// # use ra_salsa::InternId;
/// let intern_id1 = InternId::from(22_u32);
/// let intern_id2 = InternId::from(22_usize);
/// assert_eq!(intern_id1, intern_id2);
/// ```
///
/// # Converting to a u32 or usize
///
/// Normally, there should be no need to access the underlying integer
/// in a `InternId`. But if you do need to do so, you can convert to a
/// `usize` using the `as_u32` or `as_usize` methods or the `From` impls.
///
/// ```
/// # use ra_salsa::InternId;;
/// let intern_id = InternId::from(22_u32);
/// let value = u32::from(intern_id);
/// assert_eq!(value, 22);
/// ```
///
/// ## Illegal values
///
/// Be warned, however, that `InternId` values cannot be created from
/// *arbitrary* values -- in particular large values greater than
/// `InternId::MAX` will panic. Those large values are reserved so that
/// the Rust compiler can use them as sentinel values, which means
/// that (for example) `Option<InternId>` is represented in a single
/// word.
///
/// ```should_panic
/// # use ra_salsa::InternId;;
/// InternId::from(InternId::MAX);
/// ```
///
/// [rfc]: https://github.com/salsa-rs/salsa-rfcs/pull/2
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InternId {
    value: NonZeroU32,
}

impl InternId {
    /// The maximum allowed `InternId`. This value can grow between
    /// releases without affecting semver.
    pub const MAX: u32 = 0xFFFF_FF00;

    /// Creates a new InternId.
    ///
    /// # Safety
    ///
    /// `value` must be less than `MAX`
    pub const unsafe fn new_unchecked(value: u32) -> Self {
        debug_assert!(value < InternId::MAX);
        let value = unsafe { NonZeroU32::new_unchecked(value + 1) };
        InternId { value }
    }

    /// Convert this raw-id into a u32 value.
    ///
    /// ```
    /// # use ra_salsa::InternId;
    /// let intern_id = InternId::from(22_u32);
    /// let value = intern_id.as_usize();
    /// assert_eq!(value, 22);
    /// ```
    pub fn as_u32(self) -> u32 {
        self.value.get() - 1
    }

    /// Convert this raw-id into a usize value.
    ///
    /// ```
    /// # use ra_salsa::InternId;
    /// let intern_id = InternId::from(22_u32);
    /// let value = intern_id.as_usize();
    /// assert_eq!(value, 22);
    /// ```
    pub fn as_usize(self) -> usize {
        self.as_u32() as usize
    }
}

impl From<InternId> for u32 {
    fn from(raw: InternId) -> u32 {
        raw.as_u32()
    }
}

impl From<InternId> for usize {
    fn from(raw: InternId) -> usize {
        raw.as_usize()
    }
}

impl From<u32> for InternId {
    fn from(id: u32) -> InternId {
        assert!(id < InternId::MAX);
        unsafe { InternId::new_unchecked(id) }
    }
}

impl From<usize> for InternId {
    fn from(id: usize) -> InternId {
        assert!(id < (InternId::MAX as usize));
        unsafe { InternId::new_unchecked(id as u32) }
    }
}

impl fmt::Debug for InternId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_usize().fmt(f)
    }
}

impl fmt::Display for InternId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_usize().fmt(f)
    }
}
