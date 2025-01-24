#![unstable(
    feature = "temporary_niche_types",
    issue = "none",
    reason = "for core, alloc, and std internals until pattern types are further along"
)]

use crate::cmp::Ordering;
use crate::hash::{Hash, Hasher};
use crate::marker::StructuralPartialEq;
use crate::{fmt, pattern_type};

macro_rules! define_valid_range_type {
    ($(
        $(#[$m:meta])*
        $vis:vis struct $name:ident($int:ident is $pat:pat);
    )+) => {$(
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        $(#[$m])*
        $vis struct $name(pattern_type!($int is $pat));
        impl $name {
            #[inline]
            pub const fn new(val: $int) -> Option<Self> {
                #[allow(non_contiguous_range_endpoints)]
                if let $pat = val {
                    // SAFETY: just checked that the value matches the pattern
                    Some(unsafe { $name(crate::mem::transmute(val)) })
                } else {
                    None
                }
            }

            /// Constructs an instance of this type from the underlying integer
            /// primitive without checking whether its valid.
            ///
            /// # Safety
            /// Immediate language UB if `val` is not within the valid range for this
            /// type, as it violates the validity invariant.
            #[inline]
            pub const unsafe fn new_unchecked(val: $int) -> Self {
                // SAFETY: Caller promised that `val` is within the valid range.
                unsafe { crate::mem::transmute(val) }
            }

            #[inline]
            pub const fn as_inner(self) -> $int {
                // SAFETY: pattern types are always legal values of their base type
                // (Not using `.0` because that has perf regressions.)
                unsafe { crate::mem::transmute(self) }
            }
        }

        // This is required to allow matching a constant.  We don't get it from a derive
        // because the derived `PartialEq` would do a field projection, which is banned
        // by <https://github.com/rust-lang/compiler-team/issues/807>.
        impl StructuralPartialEq for $name {}

        impl Eq for $name {}

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_inner() == other.as_inner()
            }
        }

        impl Ord for $name {
            #[inline]
            fn cmp(&self, other: &Self) -> Ordering {
                Ord::cmp(&self.as_inner(), &other.as_inner())
            }
        }

        impl PartialOrd for $name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(Ord::cmp(self, other))
            }
        }

        impl Hash for $name {
            // Required method
            fn hash<H: Hasher>(&self, state: &mut H) {
                Hash::hash(&self.as_inner(), state);
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                <$int as fmt::Debug>::fmt(&self.as_inner(), f)
            }
        }
    )+};
}

define_valid_range_type! {
    pub struct Nanoseconds(u32 is 0..=999_999_999);
}

impl Nanoseconds {
    // SAFETY: 0 is within the valid range
    pub const ZERO: Self = unsafe { Nanoseconds::new_unchecked(0) };
}

#[rustc_const_unstable(feature = "const_default", issue = "143894")]
impl const Default for Nanoseconds {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

const HALF_USIZE: usize = usize::MAX >> 1;

define_valid_range_type! {
    pub struct NonZeroU8Inner(u8 is 1..);
    pub struct NonZeroU16Inner(u16 is 1..);
    pub struct NonZeroU32Inner(u32 is 1..);
    pub struct NonZeroU64Inner(u64 is 1..);
    pub struct NonZeroU128Inner(u128 is 1..);

    pub struct NonZeroI8Inner(i8 is ..0 | 1..);
    pub struct NonZeroI16Inner(i16 is ..0 | 1..);
    pub struct NonZeroI32Inner(i32 is ..0 | 1..);
    pub struct NonZeroI64Inner(i64 is ..0 | 1..);
    pub struct NonZeroI128Inner(i128 is ..0 | 1..);

    pub struct UsizeNoHighBit(usize is 0..=HALF_USIZE);
    pub struct NonZeroUsizeInner(usize is 1..);
    pub struct NonZeroIsizeInner(isize is ..0 | 1..);

    pub struct U32NotAllOnes(u32 is 0..u32::MAX);
    pub struct I32NotAllOnes(i32 is ..-1 | 0..);

    pub struct U64NotAllOnes(u64 is 0..u64::MAX);
    pub struct I64NotAllOnes(i64 is ..-1 | 0..);

    pub struct NonZeroCharInner(char is '\u{1}' ..= '\u{10ffff}');
}

pub trait NotAllOnesHelper {
    type Type;
}
pub type NotAllOnes<T> = <T as NotAllOnesHelper>::Type;
impl NotAllOnesHelper for u32 {
    type Type = U32NotAllOnes;
}
impl NotAllOnesHelper for i32 {
    type Type = I32NotAllOnes;
}
impl NotAllOnesHelper for u64 {
    type Type = U64NotAllOnes;
}
impl NotAllOnesHelper for i64 {
    type Type = I64NotAllOnes;
}

define_valid_range_type! {
    pub struct CodePointInner(u32 is 0..=0x10ffff);
}

impl CodePointInner {
    pub const ZERO: Self = CodePointInner::new(0).unwrap();
}

impl Default for CodePointInner {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}
