//! This is a densely packed error representation which is used on targets with
//! 64-bit pointers.
//!
//! (Note that `bitpacked` vs `unpacked` here has no relationship to
//! `#[repr(packed)]`, it just refers to attempting to use any available bits in
//! a more clever manner than `rustc`'s default layout algorithm would).
//!
//! Conceptually, it stores the same data as the "unpacked" equivalent we use on
//! other targets. Specifically, you can imagine it as an optimized version of
//! the following enum (which is roughly equivalent to what's stored by
//! `repr_unpacked::Repr`, e.g. `super::ErrorData<Box<Custom>>`):
//!
//! ```ignore (exposition-only)
//! enum ErrorData {
//!    Os(i32),
//!    Simple(ErrorKind),
//!    SimpleMessage(&'static SimpleMessage),
//!    Custom(Box<Custom>),
//! }
//! ```
//!
//! However, it packs this data into a 64bit non-zero value.
//!
//! This optimization not only allows `io::Error` to occupy a single pointer,
//! but improves `io::Result` as well, especially for situations like
//! `io::Result<()>` (which is now 64 bits) or `io::Result<u64>` (which is now
//! 128 bits), which are quite common.
//!
//! # Layout
//! Tagged values are 64 bits, with the 2 least significant bits used for the
//! tag. This means there are 4 "variants":
//!
//! - **Tag 0b00**: The first variant is equivalent to
//!   `ErrorData::SimpleMessage`, and holds a `&'static SimpleMessage` directly.
//!
//!   `SimpleMessage` has an alignment >= 4 (which is requested with
//!   `#[repr(align)]` and checked statically at the bottom of this file), which
//!   means every `&'static SimpleMessage` should have the both tag bits as 0,
//!   meaning its tagged and untagged representation are equivalent.
//!
//!   This means we can skip tagging it, which is necessary as this variant can
//!   be constructed from a `const fn`, which probably cannot tag pointers (or
//!   at least it would be difficult).
//!
//! - **Tag 0b01**: The other pointer variant holds the data for
//!   `ErrorData::Custom` and the remaining 62 bits are used to store a
//!   `Box<Custom>`. `Custom` also has alignment >= 4, so the bottom two bits
//!   are free to use for the tag.
//!
//!   The only important thing to note is that `ptr::wrapping_add` and
//!   `ptr::wrapping_sub` are used to tag the pointer, rather than bitwise
//!   operations. This should preserve the pointer's provenance, which would
//!   otherwise be lost.
//!
//! - **Tag 0b10**: Holds the data for `ErrorData::Os(i32)`. We store the `i32`
//!   in the pointer's most significant 32 bits, and don't use the bits `2..32`
//!   for anything. Using the top 32 bits is just to let us easily recover the
//!   `i32` code with the correct sign.
//!
//! - **Tag 0b11**: Holds the data for `ErrorData::Simple(ErrorKind)`. This
//!   stores the `ErrorKind` in the top 32 bits as well, although it doesn't
//!   occupy nearly that many. Most of the bits are unused here, but it's not
//!   like we need them for anything else yet.
//!
//! # Use of `NonNull<()>`
//!
//! Everything is stored in a `NonNull<()>`, which is odd, but actually serves a
//! purpose.
//!
//! Conceptually you might think of this more like:
//!
//! ```ignore (exposition-only)
//! union Repr {
//!     // holds integer (Simple/Os) variants, and
//!     // provides access to the tag bits.
//!     bits: NonZero<u64>,
//!     // Tag is 0, so this is stored untagged.
//!     msg: &'static SimpleMessage,
//!     // Tagged (offset) `Box<Custom>` pointer.
//!     tagged_custom: NonNull<()>,
//! }
//! ```
//!
//! But there are a few problems with this:
//!
//! 1. Union access is equivalent to a transmute, so this representation would
//!    require we transmute between integers and pointers in at least one
//!    direction, which may be UB (and even if not, it is likely harder for a
//!    compiler to reason about than explicit ptr->int operations).
//!
//! 2. Even if all fields of a union have a niche, the union itself doesn't,
//!    although this may change in the future. This would make things like
//!    `io::Result<()>` and `io::Result<usize>` larger, which defeats part of
//!    the motivation of this bitpacking.
//!
//! Storing everything in a `NonZero<usize>` (or some other integer) would be a
//! bit more traditional for pointer tagging, but it would lose provenance
//! information, couldn't be constructed from a `const fn`, and would probably
//! run into other issues as well.
//!
//! The `NonNull<()>` seems like the only alternative, even if it's fairly odd
//! to use a pointer type to store something that may hold an integer, some of
//! the time.

use core::marker::PhantomData;
use core::num::NonZeroUsize;
use core::ptr::NonNull;

use super::{Custom, ErrorData, ErrorKind, RawOsError, SimpleMessage};

// The 2 least-significant bits are used as tag.
const TAG_MASK: usize = 0b11;
const TAG_SIMPLE_MESSAGE: usize = 0b00;
const TAG_CUSTOM: usize = 0b01;
const TAG_OS: usize = 0b10;
const TAG_SIMPLE: usize = 0b11;

/// The internal representation.
///
/// See the module docs for more, this is just a way to hack in a check that we
/// indeed are not unwind-safe.
///
/// ```compile_fail,E0277
/// fn is_unwind_safe<T: core::panic::UnwindSafe>() {}
/// is_unwind_safe::<std::io::Error>();
/// ```
#[repr(transparent)]
#[rustc_insignificant_dtor]
pub(super) struct Repr(NonNull<()>, PhantomData<ErrorData<Box<Custom>>>);

// All the types `Repr` stores internally are Send + Sync, and so is it.
unsafe impl Send for Repr {}
unsafe impl Sync for Repr {}

impl Repr {
    pub(super) fn new(dat: ErrorData<Box<Custom>>) -> Self {
        match dat {
            ErrorData::Os(code) => Self::new_os(code),
            ErrorData::Simple(kind) => Self::new_simple(kind),
            ErrorData::SimpleMessage(simple_message) => Self::new_simple_message(simple_message),
            ErrorData::Custom(b) => Self::new_custom(b),
        }
    }

    pub(super) fn new_custom(b: Box<Custom>) -> Self {
        let p = Box::into_raw(b).cast::<u8>();
        // Should only be possible if an allocator handed out a pointer with
        // wrong alignment.
        debug_assert_eq!(p.addr() & TAG_MASK, 0);
        // Note: We know `TAG_CUSTOM <= size_of::<Custom>()` (static_assert at
        // end of file), and both the start and end of the expression must be
        // valid without address space wraparound due to `Box`'s semantics.
        //
        // This means it would be correct to implement this using `ptr::add`
        // (rather than `ptr::wrapping_add`), but it's unclear this would give
        // any benefit, so we just use `wrapping_add` instead.
        let tagged = p.wrapping_add(TAG_CUSTOM).cast::<()>();
        // Safety: `TAG_CUSTOM + p` is the same as `TAG_CUSTOM | p`,
        // because `p`'s alignment means it isn't allowed to have any of the
        // `TAG_BITS` set (you can verify that addition and bitwise-or are the
        // same when the operands have no bits in common using a truth table).
        //
        // Then, `TAG_CUSTOM | p` is not zero, as that would require
        // `TAG_CUSTOM` and `p` both be zero, and neither is (as `p` came from a
        // box, and `TAG_CUSTOM` just... isn't zero -- it's `0b01`). Therefore,
        // `TAG_CUSTOM + p` isn't zero and so `tagged` can't be, and the
        // `new_unchecked` is safe.
        let res = Self(unsafe { NonNull::new_unchecked(tagged) }, PhantomData);
        // quickly smoke-check we encoded the right thing (This generally will
        // only run in std's tests, unless the user uses -Zbuild-std)
        debug_assert!(matches!(res.data(), ErrorData::Custom(_)), "repr(custom) encoding failed");
        res
    }

    #[inline]
    pub(super) fn new_os(code: RawOsError) -> Self {
        let utagged = ((code as usize) << 32) | TAG_OS;
        // Safety: `TAG_OS` is not zero, so the result of the `|` is not 0.
        let res = Self(
            NonNull::without_provenance(unsafe { NonZeroUsize::new_unchecked(utagged) }),
            PhantomData,
        );
        // quickly smoke-check we encoded the right thing (This generally will
        // only run in std's tests, unless the user uses -Zbuild-std)
        debug_assert!(
            matches!(res.data(), ErrorData::Os(c) if c == code),
            "repr(os) encoding failed for {code}"
        );
        res
    }

    #[inline]
    pub(super) fn new_simple(kind: ErrorKind) -> Self {
        let utagged = ((kind as usize) << 32) | TAG_SIMPLE;
        // Safety: `TAG_SIMPLE` is not zero, so the result of the `|` is not 0.
        let res = Self(
            NonNull::without_provenance(unsafe { NonZeroUsize::new_unchecked(utagged) }),
            PhantomData,
        );
        // quickly smoke-check we encoded the right thing (This generally will
        // only run in std's tests, unless the user uses -Zbuild-std)
        debug_assert!(
            matches!(res.data(), ErrorData::Simple(k) if k == kind),
            "repr(simple) encoding failed {:?}",
            kind,
        );
        res
    }

    #[inline]
    pub(super) const fn new_simple_message(m: &'static SimpleMessage) -> Self {
        // Safety: References are never null.
        Self(unsafe { NonNull::new_unchecked(m as *const _ as *mut ()) }, PhantomData)
    }

    #[inline]
    pub(super) fn data(&self) -> ErrorData<&Custom> {
        // Safety: We're a Repr, decode_repr is fine.
        unsafe { decode_repr(self.0, |c| &*c) }
    }

    #[inline]
    pub(super) fn data_mut(&mut self) -> ErrorData<&mut Custom> {
        // Safety: We're a Repr, decode_repr is fine.
        unsafe { decode_repr(self.0, |c| &mut *c) }
    }

    #[inline]
    pub(super) fn into_data(self) -> ErrorData<Box<Custom>> {
        let this = core::mem::ManuallyDrop::new(self);
        // Safety: We're a Repr, decode_repr is fine. The `Box::from_raw` is
        // safe because we prevent double-drop using `ManuallyDrop`.
        unsafe { decode_repr(this.0, |p| Box::from_raw(p)) }
    }
}

impl Drop for Repr {
    #[inline]
    fn drop(&mut self) {
        // Safety: We're a Repr, decode_repr is fine. The `Box::from_raw` is
        // safe because we're being dropped.
        unsafe {
            let _ = decode_repr(self.0, |p| Box::<Custom>::from_raw(p));
        }
    }
}

// Shared helper to decode a `Repr`'s internal pointer into an ErrorData.
//
// Safety: `ptr`'s bits should be encoded as described in the document at the
// top (it should `some_repr.0`)
#[inline]
unsafe fn decode_repr<C, F>(ptr: NonNull<()>, make_custom: F) -> ErrorData<C>
where
    F: FnOnce(*mut Custom) -> C,
{
    let bits = ptr.as_ptr().addr();
    match bits & TAG_MASK {
        TAG_OS => {
            let code = ((bits as i64) >> 32) as RawOsError;
            ErrorData::Os(code)
        }
        TAG_SIMPLE => {
            let kind_bits = (bits >> 32) as u32;
            let kind = kind_from_prim(kind_bits).unwrap_or_else(|| {
                debug_assert!(false, "Invalid io::error::Repr bits: `Repr({:#018x})`", bits);
                // This means the `ptr` passed in was not valid, which violates
                // the unsafe contract of `decode_repr`.
                //
                // Using this rather than unwrap meaningfully improves the code
                // for callers which only care about one variant (usually
                // `Custom`)
                unsafe { core::hint::unreachable_unchecked() };
            });
            ErrorData::Simple(kind)
        }
        TAG_SIMPLE_MESSAGE => {
            // SAFETY: per tag
            unsafe { ErrorData::SimpleMessage(&*ptr.cast::<SimpleMessage>().as_ptr()) }
        }
        TAG_CUSTOM => {
            // It would be correct for us to use `ptr::byte_sub` here (see the
            // comment above the `wrapping_add` call in `new_custom` for why),
            // but it isn't clear that it makes a difference, so we don't.
            let custom = ptr.as_ptr().wrapping_byte_sub(TAG_CUSTOM).cast::<Custom>();
            ErrorData::Custom(make_custom(custom))
        }
        _ => {
            // Can't happen, and compiler can tell
            unreachable!();
        }
    }
}

// This compiles to the same code as the check+transmute, but doesn't require
// unsafe, or to hard-code max ErrorKind or its size in a way the compiler
// couldn't verify.
#[inline]
fn kind_from_prim(ek: u32) -> Option<ErrorKind> {
    macro_rules! from_prim {
        ($prim:expr => $Enum:ident { $($Variant:ident),* $(,)? }) => {{
            // Force a compile error if the list gets out of date.
            const _: fn(e: $Enum) = |e: $Enum| match e {
                $($Enum::$Variant => ()),*
            };
            match $prim {
                $(v if v == ($Enum::$Variant as _) => Some($Enum::$Variant),)*
                _ => None,
            }
        }}
    }
    from_prim!(ek => ErrorKind {
        NotFound,
        PermissionDenied,
        ConnectionRefused,
        ConnectionReset,
        HostUnreachable,
        NetworkUnreachable,
        ConnectionAborted,
        NotConnected,
        AddrInUse,
        AddrNotAvailable,
        NetworkDown,
        BrokenPipe,
        AlreadyExists,
        WouldBlock,
        NotADirectory,
        IsADirectory,
        DirectoryNotEmpty,
        ReadOnlyFilesystem,
        FilesystemLoop,
        StaleNetworkFileHandle,
        InvalidInput,
        InvalidData,
        TimedOut,
        WriteZero,
        StorageFull,
        NotSeekable,
        QuotaExceeded,
        FileTooLarge,
        ResourceBusy,
        ExecutableFileBusy,
        Deadlock,
        CrossesDevices,
        TooManyLinks,
        InvalidFilename,
        ArgumentListTooLong,
        Interrupted,
        Other,
        UnexpectedEof,
        Unsupported,
        OutOfMemory,
        InProgress,
        Uncategorized,
    })
}

// Some static checking to alert us if a change breaks any of the assumptions
// that our encoding relies on for correctness and soundness. (Some of these are
// a bit overly thorough/cautious, admittedly)
//
// If any of these are hit on a platform that std supports, we should likely
// just use `repr_unpacked.rs` there instead (unless the fix is easy).
macro_rules! static_assert {
    ($condition:expr) => {
        const _: () = assert!($condition);
    };
    (@usize_eq: $lhs:expr, $rhs:expr) => {
        const _: [(); $lhs] = [(); $rhs];
    };
}

// The bitpacking we use requires pointers be exactly 64 bits.
static_assert!(@usize_eq: size_of::<NonNull<()>>(), 8);

// We also require pointers and usize be the same size.
static_assert!(@usize_eq: size_of::<NonNull<()>>(), size_of::<usize>());

// `Custom` and `SimpleMessage` need to be thin pointers.
static_assert!(@usize_eq: size_of::<&'static SimpleMessage>(), 8);
static_assert!(@usize_eq: size_of::<Box<Custom>>(), 8);

static_assert!((TAG_MASK + 1).is_power_of_two());
// And they must have sufficient alignment.
static_assert!(align_of::<SimpleMessage>() >= TAG_MASK + 1);
static_assert!(align_of::<Custom>() >= TAG_MASK + 1);

static_assert!(@usize_eq: TAG_MASK & TAG_SIMPLE_MESSAGE, TAG_SIMPLE_MESSAGE);
static_assert!(@usize_eq: TAG_MASK & TAG_CUSTOM, TAG_CUSTOM);
static_assert!(@usize_eq: TAG_MASK & TAG_OS, TAG_OS);
static_assert!(@usize_eq: TAG_MASK & TAG_SIMPLE, TAG_SIMPLE);

// This is obviously true (`TAG_CUSTOM` is `0b01`), but in `Repr::new_custom` we
// offset a pointer by this value, and expect it to both be within the same
// object, and to not wrap around the address space. See the comment in that
// function for further details.
//
// Actually, at the moment we use `ptr::wrapping_add`, not `ptr::add`, so this
// check isn't needed for that one, although the assertion that we don't
// actually wrap around in that wrapping_add does simplify the safety reasoning
// elsewhere considerably.
static_assert!(size_of::<Custom>() >= TAG_CUSTOM);

// These two store a payload which is allowed to be zero, so they must be
// non-zero to preserve the `NonNull`'s range invariant.
static_assert!(TAG_OS != 0);
static_assert!(TAG_SIMPLE != 0);
// We can't tag `SimpleMessage`s, the tag must be 0.
static_assert!(@usize_eq: TAG_SIMPLE_MESSAGE, 0);

// Check that the point of all of this still holds.
//
// We'd check against `io::Error`, but *technically* it's allowed to vary,
// as it's not `#[repr(transparent)]`/`#[repr(C)]`. We could add that, but
// the `#[repr()]` would show up in rustdoc, which might be seen as a stable
// commitment.
static_assert!(@usize_eq: size_of::<Repr>(), 8);
static_assert!(@usize_eq: size_of::<Option<Repr>>(), 8);
static_assert!(@usize_eq: size_of::<Result<(), Repr>>(), 8);
static_assert!(@usize_eq: size_of::<Result<usize, Repr>>(), 16);
