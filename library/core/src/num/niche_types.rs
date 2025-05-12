#![unstable(
    feature = "temporary_niche_types",
    issue = "none",
    reason = "for core, alloc, and std internals until pattern types are further along"
)]

use crate::cmp::Ordering;
use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::marker::StructuralPartialEq;

macro_rules! define_valid_range_type {
    ($(
        $(#[$m:meta])*
        $vis:vis struct $name:ident($int:ident as $uint:ident in $low:literal..=$high:literal);
    )+) => {$(
        #[derive(Clone, Copy, Eq)]
        #[repr(transparent)]
        #[rustc_layout_scalar_valid_range_start($low)]
        #[rustc_layout_scalar_valid_range_end($high)]
        $(#[$m])*
        $vis struct $name($int);

        const _: () = {
            // With the `valid_range` attributes, it's always specified as unsigned
            assert!(<$uint>::MIN == 0);
            let ulow: $uint = $low;
            let uhigh: $uint = $high;
            assert!(ulow <= uhigh);

            assert!(size_of::<$int>() == size_of::<$uint>());
        };

        impl $name {
            #[inline]
            pub const fn new(val: $int) -> Option<Self> {
                if (val as $uint) >= ($low as $uint) && (val as $uint) <= ($high as $uint) {
                    // SAFETY: just checked the inclusive range
                    Some(unsafe { $name(val) })
                } else {
                    None
                }
            }

            /// Constructs an instance of this type from the underlying integer
            /// primitive without checking whether its zero.
            ///
            /// # Safety
            /// Immediate language UB if `val == 0`, as it violates the validity
            /// invariant of this type.
            #[inline]
            pub const unsafe fn new_unchecked(val: $int) -> Self {
                // SAFETY: Caller promised that `val` is non-zero.
                unsafe { $name(val) }
            }

            #[inline]
            pub const fn as_inner(self) -> $int {
                // SAFETY: This is a transparent wrapper, so unwrapping it is sound
                // (Not using `.0` due to MCP#807.)
                unsafe { crate::mem::transmute(self) }
            }
        }

        // This is required to allow matching a constant.  We don't get it from a derive
        // because the derived `PartialEq` would do a field projection, which is banned
        // by <https://github.com/rust-lang/compiler-team/issues/807>.
        impl StructuralPartialEq for $name {}

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
    pub struct Nanoseconds(u32 as u32 in 0..=999_999_999);
}

impl Nanoseconds {
    // SAFETY: 0 is within the valid range
    pub const ZERO: Self = unsafe { Nanoseconds::new_unchecked(0) };
}

impl Default for Nanoseconds {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

define_valid_range_type! {
    pub struct NonZeroU8Inner(u8 as u8 in 1..=0xff);
    pub struct NonZeroU16Inner(u16 as u16 in 1..=0xff_ff);
    pub struct NonZeroU32Inner(u32 as u32 in 1..=0xffff_ffff);
    pub struct NonZeroU64Inner(u64 as u64 in 1..=0xffffffff_ffffffff);
    pub struct NonZeroU128Inner(u128 as u128 in 1..=0xffffffffffffffff_ffffffffffffffff);

    pub struct NonZeroI8Inner(i8 as u8 in 1..=0xff);
    pub struct NonZeroI16Inner(i16 as u16 in 1..=0xff_ff);
    pub struct NonZeroI32Inner(i32 as u32 in 1..=0xffff_ffff);
    pub struct NonZeroI64Inner(i64 as u64 in 1..=0xffffffff_ffffffff);
    pub struct NonZeroI128Inner(i128 as u128 in 1..=0xffffffffffffffff_ffffffffffffffff);
}

#[cfg(target_pointer_width = "16")]
define_valid_range_type! {
    pub struct UsizeNoHighBit(usize as usize in 0..=0x7fff);
    pub struct NonZeroUsizeInner(usize as usize in 1..=0xffff);
    pub struct NonZeroIsizeInner(isize as usize in 1..=0xffff);
}
#[cfg(target_pointer_width = "32")]
define_valid_range_type! {
    pub struct UsizeNoHighBit(usize as usize in 0..=0x7fff_ffff);
    pub struct NonZeroUsizeInner(usize as usize in 1..=0xffff_ffff);
    pub struct NonZeroIsizeInner(isize as usize in 1..=0xffff_ffff);
}
#[cfg(target_pointer_width = "64")]
define_valid_range_type! {
    pub struct UsizeNoHighBit(usize as usize in 0..=0x7fff_ffff_ffff_ffff);
    pub struct NonZeroUsizeInner(usize as usize in 1..=0xffff_ffff_ffff_ffff);
    pub struct NonZeroIsizeInner(isize as usize in 1..=0xffff_ffff_ffff_ffff);
}

define_valid_range_type! {
    pub struct U32NotAllOnes(u32 as u32 in 0..=0xffff_fffe);
    pub struct I32NotAllOnes(i32 as u32 in 0..=0xffff_fffe);

    pub struct U64NotAllOnes(u64 as u64 in 0..=0xffff_ffff_ffff_fffe);
    pub struct I64NotAllOnes(i64 as u64 in 0..=0xffff_ffff_ffff_fffe);
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
