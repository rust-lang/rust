//! Manually manage memory through raw pointers.
//!
//! *[See also the pointer primitive types](pointer).*
//!
//! # Safety
//!
//! Many functions in this module take raw pointers as arguments and read from
//! or write to them. For this to be safe, these pointers must be *valid*.
//! Whether a pointer is valid depends on the operation it is used for
//! (read or write), and the extent of the memory that is accessed (i.e.,
//! how many bytes are read/written). Most functions use `*mut T` and `*const T`
//! to access only a single value, in which case the documentation omits the size
//! and implicitly assumes it to be `size_of::<T>()` bytes.
//!
//! The precise rules for validity are not determined yet. The guarantees that are
//! provided at this point are very minimal:
//!
//! * A [null] pointer is *never* valid, not even for accesses of [size zero][zst].
//! * For a pointer to be valid, it is necessary, but not always sufficient, that the pointer
//!   be *dereferenceable*: the memory range of the given size starting at the pointer must all be
//!   within the bounds of a single allocated object. Note that in Rust,
//!   every (stack-allocated) variable is considered a separate allocated object.
//! * Even for operations of [size zero][zst], the pointer must not be pointing to deallocated
//!   memory, i.e., deallocation makes pointers invalid even for zero-sized operations. However,
//!   casting any non-zero integer *literal* to a pointer is valid for zero-sized accesses, even if
//!   some memory happens to exist at that address and gets deallocated. This corresponds to writing
//!   your own allocator: allocating zero-sized objects is not very hard. The canonical way to
//!   obtain a pointer that is valid for zero-sized accesses is [`NonNull::dangling`].
//FIXME: mention `ptr::invalid` above, once it is stable.
//! * All accesses performed by functions in this module are *non-atomic* in the sense
//!   of [atomic operations] used to synchronize between threads. This means it is
//!   undefined behavior to perform two concurrent accesses to the same location from different
//!   threads unless both accesses only read from memory. Notice that this explicitly
//!   includes [`read_volatile`] and [`write_volatile`]: Volatile accesses cannot
//!   be used for inter-thread synchronization.
//! * The result of casting a reference to a pointer is valid for as long as the
//!   underlying object is live and no reference (just raw pointers) is used to
//!   access the same memory. That is, reference and pointer accesses cannot be
//!   interleaved.
//!
//! These axioms, along with careful use of [`offset`] for pointer arithmetic,
//! are enough to correctly implement many useful things in unsafe code. Stronger guarantees
//! will be provided eventually, as the [aliasing] rules are being determined. For more
//! information, see the [book] as well as the section in the reference devoted
//! to [undefined behavior][ub].
//!
//! ## Alignment
//!
//! Valid raw pointers as defined above are not necessarily properly aligned (where
//! "proper" alignment is defined by the pointee type, i.e., `*const T` must be
//! aligned to `mem::align_of::<T>()`). However, most functions require their
//! arguments to be properly aligned, and will explicitly state
//! this requirement in their documentation. Notable exceptions to this are
//! [`read_unaligned`] and [`write_unaligned`].
//!
//! When a function requires proper alignment, it does so even if the access
//! has size 0, i.e., even if memory is not actually touched. Consider using
//! [`NonNull::dangling`] in such cases.
//!
//! ## Allocated object
//!
//! For several operations, such as [`offset`] or field projections (`expr.field`), the notion of an
//! "allocated object" becomes relevant. An allocated object is a contiguous region of memory.
//! Common examples of allocated objects include stack-allocated variables (each variable is a
//! separate allocated object), heap allocations (each allocation created by the global allocator is
//! a separate allocated object), and `static` variables.
//!
//! # Strict Provenance
//!
//! **The following text is non-normative, insufficiently formal, and is an extremely strict
//! interpretation of provenance. It's ok if your code doesn't strictly conform to it.**
//!
//! [Strict Provenance][] is an experimental set of APIs that help tools that try
//! to validate the memory-safety of your program's execution. Notably this includes [Miri][]
//! and [CHERI][], which can detect when you access out of bounds memory or otherwise violate
//! Rust's memory model.
//!
//! Provenance must exist in some form for any programming
//! language compiled for modern computer architectures, but specifying a model for provenance
//! in a way that is useful to both compilers and programmers is an ongoing challenge.
//! The [Strict Provenance][] experiment seeks to explore the question: *what if we just said you
//! couldn't do all the nasty operations that make provenance so messy?*
//!
//! What APIs would have to be removed? What APIs would have to be added? How much would code
//! have to change, and is it worse or better now? Would any patterns become truly inexpressible?
//! Could we carve out special exceptions for those patterns? Should we?
//!
//! A secondary goal of this project is to see if we can disambiguate the many functions of
//! pointer<->integer casts enough for the definition of `usize` to be loosened so that it
//! isn't *pointer*-sized but address-space/offset/allocation-sized (we'll probably continue
//! to conflate these notions). This would potentially make it possible to more efficiently
//! target platforms where pointers are larger than offsets, such as CHERI and maybe some
//! segmented architectures.
//!
//! ## Provenance
//!
//! **This section is *non-normative* and is part of the [Strict Provenance][] experiment.**
//!
//! Pointers are not *simply* an "integer" or "address". For instance, it's uncontroversial
//! to say that a Use After Free is clearly Undefined Behaviour, even if you "get lucky"
//! and the freed memory gets reallocated before your read/write (in fact this is the
//! worst-case scenario, UAFs would be much less concerning if this didn't happen!).
//! To rationalize this claim, pointers need to somehow be *more* than just their addresses:
//! they must have provenance.
//!
//! When an allocation is created, that allocation has a unique Original Pointer. For alloc
//! APIs this is literally the pointer the call returns, and for local variables and statics,
//! this is the name of the variable/static. This is mildly overloading the term "pointer"
//! for the sake of brevity/exposition.
//!
//! The Original Pointer for an allocation is guaranteed to have unique access to the entire
//! allocation and *only* that allocation. In this sense, an allocation can be thought of
//! as a "sandbox" that cannot be broken into or out of. *Provenance* is the permission
//! to access an allocation's sandbox and has both a *spatial* and *temporal* component:
//!
//! * Spatial: A range of bytes that the pointer is allowed to access.
//! * Temporal: The lifetime (of the allocation) that access to these bytes is tied to.
//!
//! Spatial provenance makes sure you don't go beyond your sandbox, while temporal provenance
//! makes sure that you can't "get lucky" after your permission to access some memory
//! has been revoked (either through deallocations or borrows expiring).
//!
//! Provenance is implicitly shared with all pointers transitively derived from
//! The Original Pointer through operations like [`offset`], borrowing, and pointer casts.
//! Some operations may *shrink* the derived provenance, limiting how much memory it can
//! access or how long it's valid for (i.e. borrowing a subfield and subslicing).
//!
//! Shrinking provenance cannot be undone: even if you "know" there is a larger allocation, you
//! can't derive a pointer with a larger provenance. Similarly, you cannot "recombine"
//! two contiguous provenances back into one (i.e. with a `fn merge(&[T], &[T]) -> &[T]`).
//!
//! A reference to a value always has provenance over exactly the memory that field occupies.
//! A reference to a slice always has provenance over exactly the range that slice describes.
//!
//! If an allocation is deallocated, all pointers with provenance to that allocation become
//! invalidated, and effectively lose their provenance.
//!
//! The strict provenance experiment is mostly only interested in exploring stricter *spatial*
//! provenance. In this sense it can be thought of as a subset of the more ambitious and
//! formal [Stacked Borrows][] research project, which is what tools like [Miri][] are based on.
//! In particular, Stacked Borrows is necessary to properly describe what borrows are allowed
//! to do and when they become invalidated. This necessarily involves much more complex
//! *temporal* reasoning than simply identifying allocations. Adjusting APIs and code
//! for the strict provenance experiment will also greatly help Stacked Borrows.
//!
//!
//! ## Pointer Vs Addresses
//!
//! **This section is *non-normative* and is part of the [Strict Provenance][] experiment.**
//!
//! One of the largest historical issues with trying to define provenance is that programmers
//! freely convert between pointers and integers. Once you allow for this, it generally becomes
//! impossible to accurately track and preserve provenance information, and you need to appeal
//! to very complex and unreliable heuristics. But of course, converting between pointers and
//! integers is very useful, so what can we do?
//!
//! Also did you know WASM is actually a "Harvard Architecture"? As in function pointers are
//! handled completely differently from data pointers? And we kind of just shipped Rust on WASM
//! without really addressing the fact that we let you freely convert between function pointers
//! and data pointers, because it mostly Just Works? Let's just put that on the "pointer casts
//! are dubious" pile.
//!
//! Strict Provenance attempts to square these circles by decoupling Rust's traditional conflation
//! of pointers and `usize` (and `isize`), and defining a pointer to semantically contain the
//! following information:
//!
//! * The **address-space** it is part of (e.g. "data" vs "code" in WASM).
//! * The **address** it points to, which can be represented by a `usize`.
//! * The **provenance** it has, defining the memory it has permission to access.
//!
//! Under Strict Provenance, a usize *cannot* accurately represent a pointer, and converting from
//! a pointer to a usize is generally an operation which *only* extracts the address. It is
//! therefore *impossible* to construct a valid pointer from a usize because there is no way
//! to restore the address-space and provenance. In other words, pointer-integer-pointer
//! roundtrips are not possible (in the sense that the resulting pointer is not dereferenceable).
//!
//! The key insight to making this model *at all* viable is the [`with_addr`][] method:
//!
//! ```text
//!     /// Creates a new pointer with the given address.
//!     ///
//!     /// This performs the same operation as an `addr as ptr` cast, but copies
//!     /// the *address-space* and *provenance* of `self` to the new pointer.
//!     /// This allows us to dynamically preserve and propagate this important
//!     /// information in a way that is otherwise impossible with a unary cast.
//!     ///
//!     /// This is equivalent to using `wrapping_offset` to offset `self` to the
//!     /// given address, and therefore has all the same capabilities and restrictions.
//!     pub fn with_addr(self, addr: usize) -> Self;
//! ```
//!
//! So you're still able to drop down to the address representation and do whatever
//! clever bit tricks you want *as long as* you're able to keep around a pointer
//! into the allocation you care about that can "reconstitute" the other parts of the pointer.
//! Usually this is very easy, because you only are taking a pointer, messing with the address,
//! and then immediately converting back to a pointer. To make this use case more ergonomic,
//! we provide the [`map_addr`][] method.
//!
//! To help make it clear that code is "following" Strict Provenance semantics, we also provide an
//! [`addr`][] method which promises that the returned address is not part of a
//! pointer-usize-pointer roundtrip. In the future we may provide a lint for pointer<->integer
//! casts to help you audit if your code conforms to strict provenance.
//!
//!
//! ## Using Strict Provenance
//!
//! Most code needs no changes to conform to strict provenance, as the only really concerning
//! operation that *wasn't* obviously already Undefined Behaviour is casts from usize to a
//! pointer. For code which *does* cast a usize to a pointer, the scope of the change depends
//! on exactly what you're doing.
//!
//! In general you just need to make sure that if you want to convert a usize address to a
//! pointer and then use that pointer to read/write memory, you need to keep around a pointer
//! that has sufficient provenance to perform that read/write itself. In this way all of your
//! casts from an address to a pointer are essentially just applying offsets/indexing.
//!
//! This is generally trivial to do for simple cases like tagged pointers *as long as you
//! represent the tagged pointer as an actual pointer and not a usize*. For instance:
//!
//! ```
//! #![feature(strict_provenance)]
//!
//! unsafe {
//!     // A flag we want to pack into our pointer
//!     static HAS_DATA: usize = 0x1;
//!     static FLAG_MASK: usize = !HAS_DATA;
//!
//!     // Our value, which must have enough alignment to have spare least-significant-bits.
//!     let my_precious_data: u32 = 17;
//!     assert!(core::mem::align_of::<u32>() > 1);
//!
//!     // Create a tagged pointer
//!     let ptr = &my_precious_data as *const u32;
//!     let tagged = ptr.map_addr(|addr| addr | HAS_DATA);
//!
//!     // Check the flag:
//!     if tagged.addr() & HAS_DATA != 0 {
//!         // Untag and read the pointer
//!         let data = *tagged.map_addr(|addr| addr & FLAG_MASK);
//!         assert_eq!(data, 17);
//!     } else {
//!         unreachable!()
//!     }
//! }
//! ```
//!
//! (Yes, if you've been using AtomicUsize for pointers in concurrent datastructures, you should
//! be using AtomicPtr instead. If that messes up the way you atomically manipulate pointers,
//! we would like to know why, and what needs to be done to fix it.)
//!
//! Something more complicated and just generally *evil* like an XOR-List requires more significant
//! changes like allocating all nodes in a pre-allocated Vec or Arena and using a pointer
//! to the whole allocation to reconstitute the XORed addresses.
//!
//! Situations where a valid pointer *must* be created from just an address, such as baremetal code
//! accessing a memory-mapped interface at a fixed address, are an open question on how to support.
//! These situations *will* still be allowed, but we might require some kind of "I know what I'm
//! doing" annotation to explain the situation to the compiler. It's also possible they need no
//! special attention at all, because they're generally accessing memory outside the scope of
//! "the abstract machine", or already using "I know what I'm doing" annotations like "volatile".
//!
//! Under [Strict Provenance] it is Undefined Behaviour to:
//!
//! * Access memory through a pointer that does not have provenance over that memory.
//!
//! * [`offset`] a pointer to or from an address it doesn't have provenance over.
//!   This means it's always UB to offset a pointer derived from something deallocated,
//!   even if the offset is 0. Note that a pointer "one past the end" of its provenance
//!   is not actually outside its provenance, it just has 0 bytes it can load/store.
//!
//! But it *is* still sound to:
//!
//! * Create an invalid pointer from just an address (see [`ptr::invalid`][]). This can
//!   be used for sentinel values like `null` *or* to represent a tagged pointer that will
//!   never be dereferenceable. In general, it is always sound for an integer to pretend
//!   to be a pointer "for fun" as long as you don't use operations on it which require
//!   it to be valid (offset, read, write, etc).
//!
//! * Forge an allocation of size zero at any sufficiently aligned non-null address.
//!   i.e. the usual "ZSTs are fake, do what you want" rules apply *but* this only applies
//!   for actual forgery (integers cast to pointers). If you borrow some struct's field
//!   that *happens* to be zero-sized, the resulting pointer will have provenance tied to
//!   that allocation and it will still get invalidated if the allocation gets deallocated.
//!   In the future we may introduce an API to make such a forged allocation explicit.
//!
//! * [`wrapping_offset`][] a pointer outside its provenance. This includes invalid pointers
//!   which have "no" provenance. Unfortunately there may be practical limits on this for a
//!   particular platform, and it's an open question as to how to specify this (if at all).
//!   Notably, [CHERI][] relies on a compression scheme that can't handle a
//!   pointer getting offset "too far" out of bounds. If this happens, the address
//!   returned by `addr` will be the value you expect, but the provenance will get invalidated
//!   and using it to read/write will fault. The details of this are architecture-specific
//!   and based on alignment, but the buffer on either side of the pointer's range is pretty
//!   generous (think kilobytes, not bytes).
//!
//! * Compare arbitrary pointers by address. Addresses *are* just integers and so there is
//!   always a coherent answer, even if the pointers are invalid or from different
//!   address-spaces/provenances. Of course, comparing addresses from different address-spaces
//!   is generally going to be *meaningless*, but so is comparing Kilograms to Meters, and Rust
//!   doesn't prevent that either. Similarly, if you get "lucky" and notice that a pointer
//!   one-past-the-end is the "same" address as the start of an unrelated allocation, anything
//!   you do with that fact is *probably* going to be gibberish. The scope of that gibberish
//!   is kept under control by the fact that the two pointers *still* aren't allowed to access
//!   the other's allocation (bytes), because they still have different provenance.
//!
//! * Perform pointer tagging tricks. This falls out of [`wrapping_offset`] but is worth
//!   mentioning in more detail because of the limitations of [CHERI][]. Low-bit tagging
//!   is very robust, and often doesn't even go out of bounds because types ensure
//!   size >= align (and over-aligning actually gives CHERI more flexibility). Anything
//!   more complex than this rapidly enters "extremely platform-specific" territory as
//!   certain things may or may not be allowed based on specific supported operations.
//!   For instance, ARM explicitly supports high-bit tagging, and so CHERI on ARM inherits
//!   that and should support it.
//!
//! ## Pointer-usize-pointer roundtrips and 'exposed' provenance
//!
//! **This section is *non-normative* and is part of the [Strict Provenance] experiment.**
//!
//! As discussed above, pointer-usize-pointer roundtrips are not possible under [Strict Provenance].
//! However, there exists legacy Rust code that is full of such roundtrips, and legacy platform APIs
//! regularly assume that `usize` can capture all the information that makes up a pointer. There
//! also might be code that cannot be ported to Strict Provenance (which is something we would [like
//! to hear about][Strict Provenance]).
//!
//! For situations like this, there is a fallback plan, a way to 'opt out' of Strict Provenance.
//! However, note that this makes your code a lot harder to specify, and the code will not work
//! (well) with tools like [Miri] and [CHERI].
//!
//! This fallback plan is provided by the [`expose_addr`] and [`from_exposed_addr`] methods (which
//! are equivalent to `as` casts between pointers and integers). [`expose_addr`] is a lot like
//! [`addr`], but additionally adds the provenance of the pointer to a global list of 'exposed'
//! provenances. (This list is purely conceptual, it exists for the purpose of specifying Rust but
//! is not materialized in actual executions, except in tools like [Miri].) [`from_exposed_addr`]
//! can be used to construct a pointer with one of these previously 'exposed' provenances.
//! [`from_exposed_addr`] takes only `addr: usize` as arguments, so unlike in [`with_addr`] there is
//! no indication of what the correct provenance for the returned pointer is -- and that is exactly
//! what makes pointer-usize-pointer roundtrips so tricky to rigorously specify! There is no
//! algorithm that decides which provenance will be used. You can think of this as "guessing" the
//! right provenance, and the guess will be "maximally in your favor", in the sense that if there is
//! any way to avoid undefined behavior, then that is the guess that will be taken. However, if
//! there is *no* previously 'exposed' provenance that justifies the way the returned pointer will
//! be used, the program has undefined behavior.
//!
//! Using [`expose_addr`] or [`from_exposed_addr`] (or the equivalent `as` casts) means that code is
//! *not* following Strict Provenance rules. The goal of the Strict Provenance experiment is to
//! determine whether it is possible to use Rust without [`expose_addr`] and [`from_exposed_addr`].
//! If this is successful, it would be a major win for avoiding specification complexity and to
//! facilitate adoption of tools like [CHERI] and [Miri] that can be a big help in increasing the
//! confidence in (unsafe) Rust code.
//!
//! [aliasing]: ../../nomicon/aliasing.html
//! [book]: ../../book/ch19-01-unsafe-rust.html#dereferencing-a-raw-pointer
//! [ub]: ../../reference/behavior-considered-undefined.html
//! [zst]: ../../nomicon/exotic-sizes.html#zero-sized-types-zsts
//! [atomic operations]: crate::sync::atomic
//! [`offset`]: pointer::offset
//! [`wrapping_offset`]: pointer::wrapping_offset
//! [`with_addr`]: pointer::with_addr
//! [`map_addr`]: pointer::map_addr
//! [`addr`]: pointer::addr
//! [`ptr::invalid`]: core::ptr::invalid
//! [`expose_addr`]: pointer::expose_addr
//! [`from_exposed_addr`]: from_exposed_addr
//! [Miri]: https://github.com/rust-lang/miri
//! [CHERI]: https://www.cl.cam.ac.uk/research/security/ctsrd/cheri/
//! [Strict Provenance]: https://github.com/rust-lang/rust/issues/95228
//! [Stacked Borrows]: https://plv.mpi-sws.org/rustbelt/stacked-borrows/

#![stable(feature = "rust1", since = "1.0.0")]

use crate::cmp::Ordering;
use crate::fmt;
use crate::hash;
use crate::intrinsics::{
    self, assert_unsafe_precondition, is_aligned_and_not_null, is_nonoverlapping,
};

use crate::mem::{self, MaybeUninit};

mod alignment;
#[unstable(feature = "ptr_alignment_type", issue = "102070")]
pub use alignment::Alignment;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::copy_nonoverlapping;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::copy;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::write_bytes;

mod metadata;
#[unstable(feature = "ptr_metadata", issue = "81513")]
pub use metadata::{from_raw_parts, from_raw_parts_mut, metadata, DynMetadata, Pointee, Thin};

mod non_null;
#[stable(feature = "nonnull", since = "1.25.0")]
pub use non_null::NonNull;

mod unique;
#[unstable(feature = "ptr_internals", issue = "none")]
pub use unique::Unique;

mod const_ptr;
mod mut_ptr;

/// Executes the destructor (if any) of the pointed-to value.
///
/// This is semantically equivalent to calling [`ptr::read`] and discarding
/// the result, but has the following advantages:
///
/// * It is *required* to use `drop_in_place` to drop unsized types like
///   trait objects, because they can't be read out onto the stack and
///   dropped normally.
///
/// * It is friendlier to the optimizer to do this over [`ptr::read`] when
///   dropping manually allocated memory (e.g., in the implementations of
///   `Box`/`Rc`/`Vec`), as the compiler doesn't need to prove that it's
///   sound to elide the copy.
///
/// * It can be used to drop [pinned] data when `T` is not `repr(packed)`
///   (pinned data must not be moved before it is dropped).
///
/// Unaligned values cannot be dropped in place, they must be copied to an aligned
/// location first using [`ptr::read_unaligned`]. For packed structs, this move is
/// done automatically by the compiler. This means the fields of packed structs
/// are not dropped in-place.
///
/// [`ptr::read`]: self::read
/// [`ptr::read_unaligned`]: self::read_unaligned
/// [pinned]: crate::pin
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `to_drop` must be [valid] for both reads and writes.
///
/// * `to_drop` must be properly aligned.
///
/// * The value `to_drop` points to must be valid for dropping, which may mean it must uphold
///   additional invariants - this is type-dependent.
///
/// Additionally, if `T` is not [`Copy`], using the pointed-to value after
/// calling `drop_in_place` can cause undefined behavior. Note that `*to_drop =
/// foo` counts as a use because it will cause the value to be dropped
/// again. [`write()`] can be used to overwrite data without causing it to be
/// dropped.
///
/// Note that even if `T` has size `0`, the pointer must be non-null and properly aligned.
///
/// [valid]: self#safety
///
/// # Examples
///
/// Manually remove the last item from a vector:
///
/// ```
/// use std::ptr;
/// use std::rc::Rc;
///
/// let last = Rc::new(1);
/// let weak = Rc::downgrade(&last);
///
/// let mut v = vec![Rc::new(0), last];
///
/// unsafe {
///     // Get a raw pointer to the last element in `v`.
///     let ptr = &mut v[1] as *mut _;
///     // Shorten `v` to prevent the last item from being dropped. We do that first,
///     // to prevent issues if the `drop_in_place` below panics.
///     v.set_len(1);
///     // Without a call `drop_in_place`, the last item would never be dropped,
///     // and the memory it manages would be leaked.
///     ptr::drop_in_place(ptr);
/// }
///
/// assert_eq!(v, &[0.into()]);
///
/// // Ensure that the last item was dropped.
/// assert!(weak.upgrade().is_none());
/// ```
#[stable(feature = "drop_in_place", since = "1.8.0")]
#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.

    // SAFETY: see comment above
    unsafe { drop_in_place(to_drop) }
}

/// Creates a null raw pointer.
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let p: *const i32 = ptr::null();
/// assert!(p.is_null());
/// ```
#[inline(always)]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
#[rustc_const_stable(feature = "const_ptr_null", since = "1.24.0")]
#[rustc_allow_const_fn_unstable(ptr_metadata)]
#[rustc_diagnostic_item = "ptr_null"]
pub const fn null<T: ?Sized + Thin>() -> *const T {
    from_raw_parts(invalid(0), ())
}

/// Creates a null mutable raw pointer.
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let p: *mut i32 = ptr::null_mut();
/// assert!(p.is_null());
/// ```
#[inline(always)]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
#[rustc_const_stable(feature = "const_ptr_null", since = "1.24.0")]
#[rustc_allow_const_fn_unstable(ptr_metadata)]
#[rustc_diagnostic_item = "ptr_null_mut"]
pub const fn null_mut<T: ?Sized + Thin>() -> *mut T {
    from_raw_parts_mut(invalid_mut(0), ())
}

/// Creates an invalid pointer with the given address.
///
/// This is different from `addr as *const T`, which creates a pointer that picks up a previously
/// exposed provenance. See [`from_exposed_addr`] for more details on that operation.
///
/// The module's top-level documentation discusses the precise meaning of an "invalid"
/// pointer but essentially this expresses that the pointer is not associated
/// with any actual allocation and is little more than a usize address in disguise.
///
/// This pointer will have no provenance associated with it and is therefore
/// UB to read/write/offset. This mostly exists to facilitate things
/// like `ptr::null` and `NonNull::dangling` which make invalid pointers.
///
/// (Standard "Zero-Sized-Types get to cheat and lie" caveats apply, although it
/// may be desirable to give them their own API just to make that 100% clear.)
///
/// This API and its claimed semantics are part of the Strict Provenance experiment,
/// see the [module documentation][crate::ptr] for details.
#[inline(always)]
#[must_use]
#[rustc_const_stable(feature = "stable_things_using_strict_provenance", since = "1.61.0")]
#[unstable(feature = "strict_provenance", issue = "95228")]
pub const fn invalid<T>(addr: usize) -> *const T {
    // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
    // We use transmute rather than a cast so tools like Miri can tell that this
    // is *not* the same as from_exposed_addr.
    // SAFETY: every valid integer is also a valid pointer (as long as you don't dereference that
    // pointer).
    unsafe { mem::transmute(addr) }
}

/// Creates an invalid mutable pointer with the given address.
///
/// This is different from `addr as *mut T`, which creates a pointer that picks up a previously
/// exposed provenance. See [`from_exposed_addr_mut`] for more details on that operation.
///
/// The module's top-level documentation discusses the precise meaning of an "invalid"
/// pointer but essentially this expresses that the pointer is not associated
/// with any actual allocation and is little more than a usize address in disguise.
///
/// This pointer will have no provenance associated with it and is therefore
/// UB to read/write/offset. This mostly exists to facilitate things
/// like `ptr::null` and `NonNull::dangling` which make invalid pointers.
///
/// (Standard "Zero-Sized-Types get to cheat and lie" caveats apply, although it
/// may be desirable to give them their own API just to make that 100% clear.)
///
/// This API and its claimed semantics are part of the Strict Provenance experiment,
/// see the [module documentation][crate::ptr] for details.
#[inline(always)]
#[must_use]
#[rustc_const_stable(feature = "stable_things_using_strict_provenance", since = "1.61.0")]
#[unstable(feature = "strict_provenance", issue = "95228")]
pub const fn invalid_mut<T>(addr: usize) -> *mut T {
    // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
    // We use transmute rather than a cast so tools like Miri can tell that this
    // is *not* the same as from_exposed_addr.
    // SAFETY: every valid integer is also a valid pointer (as long as you don't dereference that
    // pointer).
    unsafe { mem::transmute(addr) }
}

/// Convert an address back to a pointer, picking up a previously 'exposed' provenance.
///
/// This is equivalent to `addr as *const T`. The provenance of the returned pointer is that of *any*
/// pointer that was previously exposed by passing it to [`expose_addr`][pointer::expose_addr],
/// or a `ptr as usize` cast. In addition, memory which is outside the control of the Rust abstract
/// machine (MMIO registers, for example) is always considered to be exposed, so long as this memory
/// is disjoint from memory that will be used by the abstract machine such as the stack, heap,
/// and statics.
///
/// If there is no 'exposed' provenance that justifies the way this pointer will be used,
/// the program has undefined behavior. In particular, the aliasing rules still apply: pointers
/// and references that have been invalidated due to aliasing accesses cannot be used any more,
/// even if they have been exposed!
///
/// Note that there is no algorithm that decides which provenance will be used. You can think of this
/// as "guessing" the right provenance, and the guess will be "maximally in your favor", in the sense
/// that if there is any way to avoid undefined behavior (while upholding all aliasing requirements),
/// then that is the guess that will be taken.
///
/// On platforms with multiple address spaces, it is your responsibility to ensure that the
/// address makes sense in the address space that this pointer will be used with.
///
/// Using this method means that code is *not* following strict provenance rules. "Guessing" a
/// suitable provenance complicates specification and reasoning and may not be supported by
/// tools that help you to stay conformant with the Rust memory model, so it is recommended to
/// use [`with_addr`][pointer::with_addr] wherever possible.
///
/// On most platforms this will produce a value with the same bytes as the address. Platforms
/// which need to store additional information in a pointer may not support this operation,
/// since it is generally not possible to actually *compute* which provenance the returned
/// pointer has to pick up.
///
/// This API and its claimed semantics are part of the Strict Provenance experiment, see the
/// [module documentation][crate::ptr] for details.
#[must_use]
#[inline(always)]
#[unstable(feature = "strict_provenance", issue = "95228")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
#[allow(fuzzy_provenance_casts)] // this *is* the strict provenance API one should use instead
pub fn from_exposed_addr<T>(addr: usize) -> *const T
where
    T: Sized,
{
    // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
    addr as *const T
}

/// Convert an address back to a mutable pointer, picking up a previously 'exposed' provenance.
///
/// This is equivalent to `addr as *mut T`. The provenance of the returned pointer is that of *any*
/// pointer that was previously passed to [`expose_addr`][pointer::expose_addr] or a `ptr as usize`
/// cast. If there is no previously 'exposed' provenance that justifies the way this pointer will be
/// used, the program has undefined behavior. Note that there is no algorithm that decides which
/// provenance will be used. You can think of this as "guessing" the right provenance, and the guess
/// will be "maximally in your favor", in the sense that if there is any way to avoid undefined
/// behavior, then that is the guess that will be taken.
///
/// On platforms with multiple address spaces, it is your responsibility to ensure that the
/// address makes sense in the address space that this pointer will be used with.
///
/// Using this method means that code is *not* following strict provenance rules. "Guessing" a
/// suitable provenance complicates specification and reasoning and may not be supported by
/// tools that help you to stay conformant with the Rust memory model, so it is recommended to
/// use [`with_addr`][pointer::with_addr] wherever possible.
///
/// On most platforms this will produce a value with the same bytes as the address. Platforms
/// which need to store additional information in a pointer may not support this operation,
/// since it is generally not possible to actually *compute* which provenance the returned
/// pointer has to pick up.
///
/// This API and its claimed semantics are part of the Strict Provenance experiment, see the
/// [module documentation][crate::ptr] for details.
#[must_use]
#[inline(always)]
#[unstable(feature = "strict_provenance", issue = "95228")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
#[allow(fuzzy_provenance_casts)] // this *is* the strict provenance API one should use instead
pub fn from_exposed_addr_mut<T>(addr: usize) -> *mut T
where
    T: Sized,
{
    // FIXME(strict_provenance_magic): I am magic and should be a compiler intrinsic.
    addr as *mut T
}

/// Convert a reference to a raw pointer.
///
/// This is equivalent to `r as *const T`, but is a bit safer since it will never silently change
/// type or mutability, in particular if the code is refactored.
#[inline(always)]
#[must_use]
#[unstable(feature = "ptr_from_ref", issue = "106116")]
pub const fn from_ref<T: ?Sized>(r: &T) -> *const T {
    r
}

/// Convert a mutable reference to a raw pointer.
///
/// This is equivalent to `r as *mut T`, but is a bit safer since it will never silently change
/// type or mutability, in particular if the code is refactored.
#[inline(always)]
#[must_use]
#[unstable(feature = "ptr_from_ref", issue = "106116")]
pub const fn from_mut<T: ?Sized>(r: &mut T) -> *mut T {
    r
}

/// Forms a raw slice from a pointer and a length.
///
/// The `len` argument is the number of **elements**, not the number of bytes.
///
/// This function is safe, but actually using the return value is unsafe.
/// See the documentation of [`slice::from_raw_parts`] for slice safety requirements.
///
/// [`slice::from_raw_parts`]: crate::slice::from_raw_parts
///
/// # Examples
///
/// ```rust
/// use std::ptr;
///
/// // create a slice pointer when starting out with a pointer to the first element
/// let x = [5, 6, 7];
/// let raw_pointer = x.as_ptr();
/// let slice = ptr::slice_from_raw_parts(raw_pointer, 3);
/// assert_eq!(unsafe { &*slice }[2], 7);
/// ```
#[inline]
#[stable(feature = "slice_from_raw_parts", since = "1.42.0")]
#[rustc_const_stable(feature = "const_slice_from_raw_parts", since = "1.64.0")]
#[rustc_allow_const_fn_unstable(ptr_metadata)]
pub const fn slice_from_raw_parts<T>(data: *const T, len: usize) -> *const [T] {
    from_raw_parts(data.cast(), len)
}

/// Performs the same functionality as [`slice_from_raw_parts`], except that a
/// raw mutable slice is returned, as opposed to a raw immutable slice.
///
/// See the documentation of [`slice_from_raw_parts`] for more details.
///
/// This function is safe, but actually using the return value is unsafe.
/// See the documentation of [`slice::from_raw_parts_mut`] for slice safety requirements.
///
/// [`slice::from_raw_parts_mut`]: crate::slice::from_raw_parts_mut
///
/// # Examples
///
/// ```rust
/// use std::ptr;
///
/// let x = &mut [5, 6, 7];
/// let raw_pointer = x.as_mut_ptr();
/// let slice = ptr::slice_from_raw_parts_mut(raw_pointer, 3);
///
/// unsafe {
///     (*slice)[2] = 99; // assign a value at an index in the slice
/// };
///
/// assert_eq!(unsafe { &*slice }[2], 99);
/// ```
#[inline]
#[stable(feature = "slice_from_raw_parts", since = "1.42.0")]
#[rustc_const_unstable(feature = "const_slice_from_raw_parts_mut", issue = "67456")]
pub const fn slice_from_raw_parts_mut<T>(data: *mut T, len: usize) -> *mut [T] {
    from_raw_parts_mut(data.cast(), len)
}

/// Swaps the values at two mutable locations of the same type, without
/// deinitializing either.
///
/// But for the following exceptions, this function is semantically
/// equivalent to [`mem::swap`]:
///
/// * It operates on raw pointers instead of references. When references are
///   available, [`mem::swap`] should be preferred.
///
/// * The two pointed-to values may overlap. If the values do overlap, then the
///   overlapping region of memory from `x` will be used. This is demonstrated
///   in the second example below.
///
/// * The operation is "untyped" in the sense that data may be uninitialized or otherwise violate
///   the requirements of `T`. The initialization state is preserved exactly.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * Both `x` and `y` must be [valid] for both reads and writes.
///
/// * Both `x` and `y` must be properly aligned.
///
/// Note that even if `T` has size `0`, the pointers must be non-null and properly aligned.
///
/// [valid]: self#safety
///
/// # Examples
///
/// Swapping two non-overlapping regions:
///
/// ```
/// use std::ptr;
///
/// let mut array = [0, 1, 2, 3];
///
/// let (x, y) = array.split_at_mut(2);
/// let x = x.as_mut_ptr().cast::<[u32; 2]>(); // this is `array[0..2]`
/// let y = y.as_mut_ptr().cast::<[u32; 2]>(); // this is `array[2..4]`
///
/// unsafe {
///     ptr::swap(x, y);
///     assert_eq!([2, 3, 0, 1], array);
/// }
/// ```
///
/// Swapping two overlapping regions:
///
/// ```
/// use std::ptr;
///
/// let mut array: [i32; 4] = [0, 1, 2, 3];
///
/// let array_ptr: *mut i32 = array.as_mut_ptr();
///
/// let x = array_ptr as *mut [i32; 3]; // this is `array[0..3]`
/// let y = unsafe { array_ptr.add(1) } as *mut [i32; 3]; // this is `array[1..4]`
///
/// unsafe {
///     ptr::swap(x, y);
///     // The indices `1..3` of the slice overlap between `x` and `y`.
///     // Reasonable results would be for to them be `[2, 3]`, so that indices `0..3` are
///     // `[1, 2, 3]` (matching `y` before the `swap`); or for them to be `[0, 1]`
///     // so that indices `1..4` are `[0, 1, 2]` (matching `x` before the `swap`).
///     // This implementation is defined to make the latter choice.
///     assert_eq!([1, 0, 1, 2], array);
/// }
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_swap", issue = "83163")]
pub const unsafe fn swap<T>(x: *mut T, y: *mut T) {
    // Give ourselves some scratch space to work with.
    // We do not have to worry about drops: `MaybeUninit` does nothing when dropped.
    let mut tmp = MaybeUninit::<T>::uninit();

    // Perform the swap
    // SAFETY: the caller must guarantee that `x` and `y` are
    // valid for writes and properly aligned. `tmp` cannot be
    // overlapping either `x` or `y` because `tmp` was just allocated
    // on the stack as a separate allocated object.
    unsafe {
        copy_nonoverlapping(x, tmp.as_mut_ptr(), 1);
        copy(y, x, 1); // `x` and `y` may overlap
        copy_nonoverlapping(tmp.as_ptr(), y, 1);
    }
}

/// Swaps `count * size_of::<T>()` bytes between the two regions of memory
/// beginning at `x` and `y`. The two regions must *not* overlap.
///
/// The operation is "untyped" in the sense that data may be uninitialized or otherwise violate the
/// requirements of `T`. The initialization state is preserved exactly.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * Both `x` and `y` must be [valid] for both reads and writes of `count *
///   size_of::<T>()` bytes.
///
/// * Both `x` and `y` must be properly aligned.
///
/// * The region of memory beginning at `x` with a size of `count *
///   size_of::<T>()` bytes must *not* overlap with the region of memory
///   beginning at `y` with the same size.
///
/// Note that even if the effectively copied size (`count * size_of::<T>()`) is `0`,
/// the pointers must be non-null and properly aligned.
///
/// [valid]: self#safety
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::ptr;
///
/// let mut x = [1, 2, 3, 4];
/// let mut y = [7, 8, 9];
///
/// unsafe {
///     ptr::swap_nonoverlapping(x.as_mut_ptr(), y.as_mut_ptr(), 2);
/// }
///
/// assert_eq!(x, [7, 8, 3, 4]);
/// assert_eq!(y, [1, 2, 9]);
/// ```
#[inline]
#[stable(feature = "swap_nonoverlapping", since = "1.27.0")]
#[rustc_const_unstable(feature = "const_swap", issue = "83163")]
pub const unsafe fn swap_nonoverlapping<T>(x: *mut T, y: *mut T, count: usize) {
    #[allow(unused)]
    macro_rules! attempt_swap_as_chunks {
        ($ChunkTy:ty) => {
            if mem::align_of::<T>() >= mem::align_of::<$ChunkTy>()
                && mem::size_of::<T>() % mem::size_of::<$ChunkTy>() == 0
            {
                let x: *mut $ChunkTy = x.cast();
                let y: *mut $ChunkTy = y.cast();
                let count = count * (mem::size_of::<T>() / mem::size_of::<$ChunkTy>());
                // SAFETY: these are the same bytes that the caller promised were
                // ok, just typed as `MaybeUninit<ChunkTy>`s instead of as `T`s.
                // The `if` condition above ensures that we're not violating
                // alignment requirements, and that the division is exact so
                // that we don't lose any bytes off the end.
                return unsafe { swap_nonoverlapping_simple_untyped(x, y, count) };
            }
        };
    }

    // SAFETY: the caller must guarantee that `x` and `y` are
    // valid for writes and properly aligned.
    unsafe {
        assert_unsafe_precondition!(
            "ptr::swap_nonoverlapping requires that both pointer arguments are aligned and non-null \
            and the specified memory ranges do not overlap",
            [T](x: *mut T, y: *mut T, count: usize) =>
            is_aligned_and_not_null(x)
                && is_aligned_and_not_null(y)
                && is_nonoverlapping(x, y, count)
        );
    }

    // Split up the slice into small power-of-two-sized chunks that LLVM is able
    // to vectorize (unless it's a special type with more-than-pointer alignment,
    // because we don't want to pessimize things like slices of SIMD vectors.)
    if mem::align_of::<T>() <= mem::size_of::<usize>()
        && (!mem::size_of::<T>().is_power_of_two()
            || mem::size_of::<T>() > mem::size_of::<usize>() * 2)
    {
        attempt_swap_as_chunks!(usize);
        attempt_swap_as_chunks!(u8);
    }

    // SAFETY: Same preconditions as this function
    unsafe { swap_nonoverlapping_simple_untyped(x, y, count) }
}

/// Same behaviour and safety conditions as [`swap_nonoverlapping`]
///
/// LLVM can vectorize this (at least it can for the power-of-two-sized types
/// `swap_nonoverlapping` tries to use) so no need to manually SIMD it.
#[inline]
#[rustc_const_unstable(feature = "const_swap", issue = "83163")]
const unsafe fn swap_nonoverlapping_simple_untyped<T>(x: *mut T, y: *mut T, count: usize) {
    let x = x.cast::<MaybeUninit<T>>();
    let y = y.cast::<MaybeUninit<T>>();
    let mut i = 0;
    while i < count {
        // SAFETY: By precondition, `i` is in-bounds because it's below `n`
        let x = unsafe { &mut *x.add(i) };
        // SAFETY: By precondition, `i` is in-bounds because it's below `n`
        // and it's distinct from `x` since the ranges are non-overlapping
        let y = unsafe { &mut *y.add(i) };
        mem::swap_simple::<MaybeUninit<T>>(x, y);

        i += 1;
    }
}

/// Moves `src` into the pointed `dst`, returning the previous `dst` value.
///
/// Neither value is dropped.
///
/// This function is semantically equivalent to [`mem::replace`] except that it
/// operates on raw pointers instead of references. When references are
/// available, [`mem::replace`] should be preferred.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for both reads and writes.
///
/// * `dst` must be properly aligned.
///
/// * `dst` must point to a properly initialized value of type `T`.
///
/// Note that even if `T` has size `0`, the pointer must be non-null and properly aligned.
///
/// [valid]: self#safety
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let mut rust = vec!['b', 'u', 's', 't'];
///
/// // `mem::replace` would have the same effect without requiring the unsafe
/// // block.
/// let b = unsafe {
///     ptr::replace(&mut rust[0], 'r')
/// };
///
/// assert_eq!(b, 'b');
/// assert_eq!(rust, &['r', 'u', 's', 't']);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_replace", issue = "83164")]
pub const unsafe fn replace<T>(dst: *mut T, mut src: T) -> T {
    // SAFETY: the caller must guarantee that `dst` is valid to be
    // cast to a mutable reference (valid for writes, aligned, initialized),
    // and cannot overlap `src` since `dst` must point to a distinct
    // allocated object.
    unsafe {
        assert_unsafe_precondition!(
            "ptr::replace requires that the pointer argument is aligned and non-null",
            [T](dst: *mut T) => is_aligned_and_not_null(dst)
        );
        mem::swap(&mut *dst, &mut src); // cannot overlap
    }
    src
}

/// Reads the value from `src` without moving it. This leaves the
/// memory in `src` unchanged.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads.
///
/// * `src` must be properly aligned. Use [`read_unaligned`] if this is not the
///   case.
///
/// * `src` must point to a properly initialized value of type `T`.
///
/// Note that even if `T` has size `0`, the pointer must be non-null and properly aligned.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let x = 12;
/// let y = &x as *const i32;
///
/// unsafe {
///     assert_eq!(std::ptr::read(y), 12);
/// }
/// ```
///
/// Manually implement [`mem::swap`]:
///
/// ```
/// use std::ptr;
///
/// fn swap<T>(a: &mut T, b: &mut T) {
///     unsafe {
///         // Create a bitwise copy of the value at `a` in `tmp`.
///         let tmp = ptr::read(a);
///
///         // Exiting at this point (either by explicitly returning or by
///         // calling a function which panics) would cause the value in `tmp` to
///         // be dropped while the same value is still referenced by `a`. This
///         // could trigger undefined behavior if `T` is not `Copy`.
///
///         // Create a bitwise copy of the value at `b` in `a`.
///         // This is safe because mutable references cannot alias.
///         ptr::copy_nonoverlapping(b, a, 1);
///
///         // As above, exiting here could trigger undefined behavior because
///         // the same value is referenced by `a` and `b`.
///
///         // Move `tmp` into `b`.
///         ptr::write(b, tmp);
///
///         // `tmp` has been moved (`write` takes ownership of its second argument),
///         // so nothing is dropped implicitly here.
///     }
/// }
///
/// let mut foo = "foo".to_owned();
/// let mut bar = "bar".to_owned();
///
/// swap(&mut foo, &mut bar);
///
/// assert_eq!(foo, "bar");
/// assert_eq!(bar, "foo");
/// ```
///
/// ## Ownership of the Returned Value
///
/// `read` creates a bitwise copy of `T`, regardless of whether `T` is [`Copy`].
/// If `T` is not [`Copy`], using both the returned value and the value at
/// `*src` can violate memory safety. Note that assigning to `*src` counts as a
/// use because it will attempt to drop the value at `*src`.
///
/// [`write()`] can be used to overwrite data without causing it to be dropped.
///
/// ```
/// use std::ptr;
///
/// let mut s = String::from("foo");
/// unsafe {
///     // `s2` now points to the same underlying memory as `s`.
///     let mut s2: String = ptr::read(&s);
///
///     assert_eq!(s2, "foo");
///
///     // Assigning to `s2` causes its original value to be dropped. Beyond
///     // this point, `s` must no longer be used, as the underlying memory has
///     // been freed.
///     s2 = String::default();
///     assert_eq!(s2, "");
///
///     // Assigning to `s` would cause the old value to be dropped again,
///     // resulting in undefined behavior.
///     // s = String::from("bar"); // ERROR
///
///     // `ptr::write` can be used to overwrite a value without dropping it.
///     ptr::write(&mut s, String::from("bar"));
/// }
///
/// assert_eq!(s, "bar");
/// ```
///
/// [valid]: self#safety
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_ptr_read", issue = "80377")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub const unsafe fn read<T>(src: *const T) -> T {
    // We are calling the intrinsics directly to avoid function calls in the generated code
    // as `intrinsics::copy_nonoverlapping` is a wrapper function.
    extern "rust-intrinsic" {
        #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
        fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
    }

    let mut tmp = MaybeUninit::<T>::uninit();
    // SAFETY: the caller must guarantee that `src` is valid for reads.
    // `src` cannot overlap `tmp` because `tmp` was just allocated on
    // the stack as a separate allocated object.
    //
    // Also, since we just wrote a valid value into `tmp`, it is guaranteed
    // to be properly initialized.
    unsafe {
        assert_unsafe_precondition!(
            "ptr::read requires that the pointer argument is aligned and non-null",
            [T](src: *const T) => is_aligned_and_not_null(src)
        );
        copy_nonoverlapping(src, tmp.as_mut_ptr(), 1);
        tmp.assume_init()
    }
}

/// Reads the value from `src` without moving it. This leaves the
/// memory in `src` unchanged.
///
/// Unlike [`read`], `read_unaligned` works with unaligned pointers.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads.
///
/// * `src` must point to a properly initialized value of type `T`.
///
/// Like [`read`], `read_unaligned` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the returned
/// value and the value at `*src` can [violate memory safety][read-ownership].
///
/// Note that even if `T` has size `0`, the pointer must be non-null.
///
/// [read-ownership]: read#ownership-of-the-returned-value
/// [valid]: self#safety
///
/// ## On `packed` structs
///
/// Attempting to create a raw pointer to an `unaligned` struct field with
/// an expression such as `&packed.unaligned as *const FieldType` creates an
/// intermediate unaligned reference before converting that to a raw pointer.
/// That this reference is temporary and immediately cast is inconsequential
/// as the compiler always expects references to be properly aligned.
/// As a result, using `&packed.unaligned as *const FieldType` causes immediate
/// *undefined behavior* in your program.
///
/// Instead you must use the [`ptr::addr_of!`](addr_of) macro to
/// create the pointer. You may use that returned pointer together with this
/// function.
///
/// An example of what not to do and how this relates to `read_unaligned` is:
///
/// ```
/// #[repr(packed, C)]
/// struct Packed {
///     _padding: u8,
///     unaligned: u32,
/// }
///
/// let packed = Packed {
///     _padding: 0x00,
///     unaligned: 0x01020304,
/// };
///
/// // Take the address of a 32-bit integer which is not aligned.
/// // In contrast to `&packed.unaligned as *const _`, this has no undefined behavior.
/// let unaligned = std::ptr::addr_of!(packed.unaligned);
///
/// let v = unsafe { std::ptr::read_unaligned(unaligned) };
/// assert_eq!(v, 0x01020304);
/// ```
///
/// Accessing unaligned fields directly with e.g. `packed.unaligned` is safe however.
///
/// # Examples
///
/// Read a usize value from a byte buffer:
///
/// ```
/// use std::mem;
///
/// fn read_usize(x: &[u8]) -> usize {
///     assert!(x.len() >= mem::size_of::<usize>());
///
///     let ptr = x.as_ptr() as *const usize;
///
///     unsafe { ptr.read_unaligned() }
/// }
/// ```
#[inline]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
#[rustc_const_unstable(feature = "const_ptr_read", issue = "80377")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub const unsafe fn read_unaligned<T>(src: *const T) -> T {
    let mut tmp = MaybeUninit::<T>::uninit();
    // SAFETY: the caller must guarantee that `src` is valid for reads.
    // `src` cannot overlap `tmp` because `tmp` was just allocated on
    // the stack as a separate allocated object.
    //
    // Also, since we just wrote a valid value into `tmp`, it is guaranteed
    // to be properly initialized.
    unsafe {
        copy_nonoverlapping(src as *const u8, tmp.as_mut_ptr() as *mut u8, mem::size_of::<T>());
        tmp.assume_init()
    }
}

/// Overwrites a memory location with the given value without reading or
/// dropping the old value.
///
/// `write` does not drop the contents of `dst`. This is safe, but it could leak
/// allocations or resources, so care should be taken not to overwrite an object
/// that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// This is appropriate for initializing uninitialized memory, or overwriting
/// memory that has previously been [`read`] from.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// * `dst` must be properly aligned. Use [`write_unaligned`] if this is not the
///   case.
///
/// Note that even if `T` has size `0`, the pointer must be non-null and properly aligned.
///
/// [valid]: self#safety
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let mut x = 0;
/// let y = &mut x as *mut i32;
/// let z = 12;
///
/// unsafe {
///     std::ptr::write(y, z);
///     assert_eq!(std::ptr::read(y), 12);
/// }
/// ```
///
/// Manually implement [`mem::swap`]:
///
/// ```
/// use std::ptr;
///
/// fn swap<T>(a: &mut T, b: &mut T) {
///     unsafe {
///         // Create a bitwise copy of the value at `a` in `tmp`.
///         let tmp = ptr::read(a);
///
///         // Exiting at this point (either by explicitly returning or by
///         // calling a function which panics) would cause the value in `tmp` to
///         // be dropped while the same value is still referenced by `a`. This
///         // could trigger undefined behavior if `T` is not `Copy`.
///
///         // Create a bitwise copy of the value at `b` in `a`.
///         // This is safe because mutable references cannot alias.
///         ptr::copy_nonoverlapping(b, a, 1);
///
///         // As above, exiting here could trigger undefined behavior because
///         // the same value is referenced by `a` and `b`.
///
///         // Move `tmp` into `b`.
///         ptr::write(b, tmp);
///
///         // `tmp` has been moved (`write` takes ownership of its second argument),
///         // so nothing is dropped implicitly here.
///     }
/// }
///
/// let mut foo = "foo".to_owned();
/// let mut bar = "bar".to_owned();
///
/// swap(&mut foo, &mut bar);
///
/// assert_eq!(foo, "bar");
/// assert_eq!(bar, "foo");
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_ptr_write", issue = "86302")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub const unsafe fn write<T>(dst: *mut T, src: T) {
    // We are calling the intrinsics directly to avoid function calls in the generated code
    // as `intrinsics::copy_nonoverlapping` is a wrapper function.
    extern "rust-intrinsic" {
        #[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
        fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
    }

    // SAFETY: the caller must guarantee that `dst` is valid for writes.
    // `dst` cannot overlap `src` because the caller has mutable access
    // to `dst` while `src` is owned by this function.
    unsafe {
        assert_unsafe_precondition!(
            "ptr::write requires that the pointer argument is aligned and non-null",
            [T](dst: *mut T) => is_aligned_and_not_null(dst)
        );
        copy_nonoverlapping(&src as *const T, dst, 1);
        intrinsics::forget(src);
    }
}

/// Overwrites a memory location with the given value without reading or
/// dropping the old value.
///
/// Unlike [`write()`], the pointer may be unaligned.
///
/// `write_unaligned` does not drop the contents of `dst`. This is safe, but it
/// could leak allocations or resources, so care should be taken not to overwrite
/// an object that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// This is appropriate for initializing uninitialized memory, or overwriting
/// memory that has previously been read with [`read_unaligned`].
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// Note that even if `T` has size `0`, the pointer must be non-null.
///
/// [valid]: self#safety
///
/// ## On `packed` structs
///
/// Attempting to create a raw pointer to an `unaligned` struct field with
/// an expression such as `&packed.unaligned as *const FieldType` creates an
/// intermediate unaligned reference before converting that to a raw pointer.
/// That this reference is temporary and immediately cast is inconsequential
/// as the compiler always expects references to be properly aligned.
/// As a result, using `&packed.unaligned as *const FieldType` causes immediate
/// *undefined behavior* in your program.
///
/// Instead you must use the [`ptr::addr_of_mut!`](addr_of_mut)
/// macro to create the pointer. You may use that returned pointer together with
/// this function.
///
/// An example of how to do it and how this relates to `write_unaligned` is:
///
/// ```
/// #[repr(packed, C)]
/// struct Packed {
///     _padding: u8,
///     unaligned: u32,
/// }
///
/// let mut packed: Packed = unsafe { std::mem::zeroed() };
///
/// // Take the address of a 32-bit integer which is not aligned.
/// // In contrast to `&packed.unaligned as *mut _`, this has no undefined behavior.
/// let unaligned = std::ptr::addr_of_mut!(packed.unaligned);
///
/// unsafe { std::ptr::write_unaligned(unaligned, 42) };
///
/// assert_eq!({packed.unaligned}, 42); // `{...}` forces copying the field instead of creating a reference.
/// ```
///
/// Accessing unaligned fields directly with e.g. `packed.unaligned` is safe however
/// (as can be seen in the `assert_eq!` above).
///
/// # Examples
///
/// Write a usize value to a byte buffer:
///
/// ```
/// use std::mem;
///
/// fn write_usize(x: &mut [u8], val: usize) {
///     assert!(x.len() >= mem::size_of::<usize>());
///
///     let ptr = x.as_mut_ptr() as *mut usize;
///
///     unsafe { ptr.write_unaligned(val) }
/// }
/// ```
#[inline]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
#[rustc_const_unstable(feature = "const_ptr_write", issue = "86302")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub const unsafe fn write_unaligned<T>(dst: *mut T, src: T) {
    // SAFETY: the caller must guarantee that `dst` is valid for writes.
    // `dst` cannot overlap `src` because the caller has mutable access
    // to `dst` while `src` is owned by this function.
    unsafe {
        copy_nonoverlapping(&src as *const T as *const u8, dst as *mut u8, mem::size_of::<T>());
        // We are calling the intrinsic directly to avoid function calls in the generated code.
        intrinsics::forget(src);
    }
}

/// Performs a volatile read of the value from `src` without moving it. This
/// leaves the memory in `src` unchanged.
///
/// Volatile operations are intended to act on I/O memory, and are guaranteed
/// to not be elided or reordered by the compiler across other volatile
/// operations.
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// The compiler shouldn't change the relative order or number of volatile
/// memory operations. However, volatile memory operations on zero-sized types
/// (e.g., if a zero-sized type is passed to `read_volatile`) are noops
/// and may be ignored.
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads.
///
/// * `src` must be properly aligned.
///
/// * `src` must point to a properly initialized value of type `T`.
///
/// Like [`read`], `read_volatile` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the returned
/// value and the value at `*src` can [violate memory safety][read-ownership].
/// However, storing non-[`Copy`] types in volatile memory is almost certainly
/// incorrect.
///
/// Note that even if `T` has size `0`, the pointer must be non-null and properly aligned.
///
/// [valid]: self#safety
/// [read-ownership]: read#ownership-of-the-returned-value
///
/// Just like in C, whether an operation is volatile has no bearing whatsoever
/// on questions involving concurrent access from multiple threads. Volatile
/// accesses behave exactly like non-atomic accesses in that regard. In particular,
/// a race between a `read_volatile` and any write operation to the same location
/// is undefined behavior.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let x = 12;
/// let y = &x as *const i32;
///
/// unsafe {
///     assert_eq!(std::ptr::read_volatile(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "volatile", since = "1.9.0")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn read_volatile<T>(src: *const T) -> T {
    // SAFETY: the caller must uphold the safety contract for `volatile_load`.
    unsafe {
        assert_unsafe_precondition!(
            "ptr::read_volatile requires that the pointer argument is aligned and non-null",
            [T](src: *const T) => is_aligned_and_not_null(src)
        );
        intrinsics::volatile_load(src)
    }
}

/// Performs a volatile write of a memory location with the given value without
/// reading or dropping the old value.
///
/// Volatile operations are intended to act on I/O memory, and are guaranteed
/// to not be elided or reordered by the compiler across other volatile
/// operations.
///
/// `write_volatile` does not drop the contents of `dst`. This is safe, but it
/// could leak allocations or resources, so care should be taken not to overwrite
/// an object that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// The compiler shouldn't change the relative order or number of volatile
/// memory operations. However, volatile memory operations on zero-sized types
/// (e.g., if a zero-sized type is passed to `write_volatile`) are noops
/// and may be ignored.
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// * `dst` must be properly aligned.
///
/// Note that even if `T` has size `0`, the pointer must be non-null and properly aligned.
///
/// [valid]: self#safety
///
/// Just like in C, whether an operation is volatile has no bearing whatsoever
/// on questions involving concurrent access from multiple threads. Volatile
/// accesses behave exactly like non-atomic accesses in that regard. In particular,
/// a race between a `write_volatile` and any other operation (reading or writing)
/// on the same location is undefined behavior.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let mut x = 0;
/// let y = &mut x as *mut i32;
/// let z = 12;
///
/// unsafe {
///     std::ptr::write_volatile(y, z);
///     assert_eq!(std::ptr::read_volatile(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "volatile", since = "1.9.0")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn write_volatile<T>(dst: *mut T, src: T) {
    // SAFETY: the caller must uphold the safety contract for `volatile_store`.
    unsafe {
        assert_unsafe_precondition!(
            "ptr::write_volatile requires that the pointer argument is aligned and non-null",
            [T](dst: *mut T) => is_aligned_and_not_null(dst)
        );
        intrinsics::volatile_store(dst, src);
    }
}

/// Align pointer `p`.
///
/// Calculate offset (in terms of elements of `size_of::<T>()` stride) that has to be applied
/// to pointer `p` so that pointer `p` would get aligned to `a`.
///
/// # Safety
/// `a` must be a power of two.
///
/// # Notes
/// This implementation has been carefully tailored to not panic. It is UB for this to panic.
/// The only real change that can be made here is change of `INV_TABLE_MOD_16` and associated
/// constants.
///
/// If we ever decide to make it possible to call the intrinsic with `a` that is not a
/// power-of-two, it will probably be more prudent to just change to a naive implementation rather
/// than trying to adapt this to accommodate that change.
///
/// Any questions go to @nagisa.
#[lang = "align_offset"]
pub(crate) const unsafe fn align_offset<T: Sized>(p: *const T, a: usize) -> usize {
    // FIXME(#75598): Direct use of these intrinsics improves codegen significantly at opt-level <=
    // 1, where the method versions of these operations are not inlined.
    use intrinsics::{
        cttz_nonzero, exact_div, mul_with_overflow, unchecked_rem, unchecked_shl, unchecked_shr,
        unchecked_sub, wrapping_add, wrapping_mul, wrapping_sub,
    };

    /// Calculate multiplicative modular inverse of `x` modulo `m`.
    ///
    /// This implementation is tailored for `align_offset` and has following preconditions:
    ///
    /// * `m` is a power-of-two;
    /// * `x < m`; (if `x ≥ m`, pass in `x % m` instead)
    ///
    /// Implementation of this function shall not panic. Ever.
    #[inline]
    const unsafe fn mod_inv(x: usize, m: usize) -> usize {
        /// Multiplicative modular inverse table modulo 2⁴ = 16.
        ///
        /// Note, that this table does not contain values where inverse does not exist (i.e., for
        /// `0⁻¹ mod 16`, `2⁻¹ mod 16`, etc.)
        const INV_TABLE_MOD_16: [u8; 8] = [1, 11, 13, 7, 9, 3, 5, 15];
        /// Modulo for which the `INV_TABLE_MOD_16` is intended.
        const INV_TABLE_MOD: usize = 16;

        // SAFETY: `m` is required to be a power-of-two, hence non-zero.
        let m_minus_one = unsafe { unchecked_sub(m, 1) };
        let mut inverse = INV_TABLE_MOD_16[(x & (INV_TABLE_MOD - 1)) >> 1] as usize;
        let mut mod_gate = INV_TABLE_MOD;
        // We iterate "up" using the following formula:
        //
        // $$ xy ≡ 1 (mod 2ⁿ) → xy (2 - xy) ≡ 1 (mod 2²ⁿ) $$
        //
        // This application needs to be applied at least until `2²ⁿ ≥ m`, at which point we can
        // finally reduce the computation to our desired `m` by taking `inverse mod m`.
        //
        // This computation is `O(log log m)`, which is to say, that on 64-bit machines this loop
        // will always finish in at most 4 iterations.
        loop {
            // y = y * (2 - xy) mod n
            //
            // Note, that we use wrapping operations here intentionally – the original formula
            // uses e.g., subtraction `mod n`. It is entirely fine to do them `mod
            // usize::MAX` instead, because we take the result `mod n` at the end
            // anyway.
            if mod_gate >= m {
                break;
            }
            inverse = wrapping_mul(inverse, wrapping_sub(2usize, wrapping_mul(x, inverse)));
            let (new_gate, overflow) = mul_with_overflow(mod_gate, mod_gate);
            if overflow {
                break;
            }
            mod_gate = new_gate;
        }
        inverse & m_minus_one
    }

    let stride = mem::size_of::<T>();

    // SAFETY: This is just an inlined `p.addr()` (which is not
    // a `const fn` so we cannot call it).
    // During const eval, we hook this function to ensure that the pointer never
    // has provenance, making this sound.
    let addr: usize = unsafe { mem::transmute(p) };

    // SAFETY: `a` is a power-of-two, therefore non-zero.
    let a_minus_one = unsafe { unchecked_sub(a, 1) };

    if stride == 0 {
        // SPECIAL_CASE: handle 0-sized types. No matter how many times we step, the address will
        // stay the same, so no offset will be able to align the pointer unless it is already
        // aligned. This branch _will_ be optimized out as `stride` is known at compile-time.
        let p_mod_a = addr & a_minus_one;
        return if p_mod_a == 0 { 0 } else { usize::MAX };
    }

    // SAFETY: `stride == 0` case has been handled by the special case above.
    let a_mod_stride = unsafe { unchecked_rem(a, stride) };
    if a_mod_stride == 0 {
        // SPECIAL_CASE: In cases where the `a` is divisible by `stride`, byte offset to align a
        // pointer can be computed more simply through `-p (mod a)`. In the off-chance the byte
        // offset is not a multiple of `stride`, the input pointer was misaligned and no pointer
        // offset will be able to produce a `p` aligned to the specified `a`.
        //
        // The naive `-p (mod a)` equation inhibits LLVM's ability to select instructions
        // like `lea`. We compute `(round_up_to_next_alignment(p, a) - p)` instead. This
        // redistributes operations around the load-bearing, but pessimizing `and` instruction
        // sufficiently for LLVM to be able to utilize the various optimizations it knows about.
        //
        // LLVM handles the branch here particularly nicely. If this branch needs to be evaluated
        // at runtime, it will produce a mask `if addr_mod_stride == 0 { 0 } else { usize::MAX }`
        // in a branch-free way and then bitwise-OR it with whatever result the `-p mod a`
        // computation produces.

        // SAFETY: `stride == 0` case has been handled by the special case above.
        let addr_mod_stride = unsafe { unchecked_rem(addr, stride) };

        return if addr_mod_stride == 0 {
            let aligned_address = wrapping_add(addr, a_minus_one) & wrapping_sub(0, a);
            let byte_offset = wrapping_sub(aligned_address, addr);
            // SAFETY: `stride` is non-zero. This is guaranteed to divide exactly as well, because
            // addr has been verified to be aligned to the original type’s alignment requirements.
            unsafe { exact_div(byte_offset, stride) }
        } else {
            usize::MAX
        };
    }

    // GENERAL_CASE: From here on we’re handling the very general case where `addr` may be
    // misaligned, there isn’t an obvious relationship between `stride` and `a` that we can take an
    // advantage of, etc. This case produces machine code that isn’t particularly high quality,
    // compared to the special cases above. The code produced here is still within the realm of
    // miracles, given the situations this case has to deal with.

    // SAFETY: a is power-of-two hence non-zero. stride == 0 case is handled above.
    let gcdpow = unsafe { cttz_nonzero(stride).min(cttz_nonzero(a)) };
    // SAFETY: gcdpow has an upper-bound that’s at most the number of bits in a usize.
    let gcd = unsafe { unchecked_shl(1usize, gcdpow) };
    // SAFETY: gcd is always greater or equal to 1.
    if addr & unsafe { unchecked_sub(gcd, 1) } == 0 {
        // This branch solves for the following linear congruence equation:
        //
        // ` p + so = 0 mod a `
        //
        // `p` here is the pointer value, `s` - stride of `T`, `o` offset in `T`s, and `a` - the
        // requested alignment.
        //
        // With `g = gcd(a, s)`, and the above condition asserting that `p` is also divisible by
        // `g`, we can denote `a' = a/g`, `s' = s/g`, `p' = p/g`, then this becomes equivalent to:
        //
        // ` p' + s'o = 0 mod a' `
        // ` o = (a' - (p' mod a')) * (s'^-1 mod a') `
        //
        // The first term is "the relative alignment of `p` to `a`" (divided by the `g`), the
        // second term is "how does incrementing `p` by `s` bytes change the relative alignment of
        // `p`" (again divided by `g`). Division by `g` is necessary to make the inverse well
        // formed if `a` and `s` are not co-prime.
        //
        // Furthermore, the result produced by this solution is not "minimal", so it is necessary
        // to take the result `o mod lcm(s, a)`. This `lcm(s, a)` is the same as `a'`.

        // SAFETY: `gcdpow` has an upper-bound not greater than the number of trailing 0-bits in
        // `a`.
        let a2 = unsafe { unchecked_shr(a, gcdpow) };
        // SAFETY: `a2` is non-zero. Shifting `a` by `gcdpow` cannot shift out any of the set bits
        // in `a` (of which it has exactly one).
        let a2minus1 = unsafe { unchecked_sub(a2, 1) };
        // SAFETY: `gcdpow` has an upper-bound not greater than the number of trailing 0-bits in
        // `a`.
        let s2 = unsafe { unchecked_shr(stride & a_minus_one, gcdpow) };
        // SAFETY: `gcdpow` has an upper-bound not greater than the number of trailing 0-bits in
        // `a`. Furthermore, the subtraction cannot overflow, because `a2 = a >> gcdpow` will
        // always be strictly greater than `(p % a) >> gcdpow`.
        let minusp2 = unsafe { unchecked_sub(a2, unchecked_shr(addr & a_minus_one, gcdpow)) };
        // SAFETY: `a2` is a power-of-two, as proven above. `s2` is strictly less than `a2`
        // because `(s % a) >> gcdpow` is strictly less than `a >> gcdpow`.
        return wrapping_mul(minusp2, unsafe { mod_inv(s2, a2) }) & a2minus1;
    }

    // Cannot be aligned at all.
    usize::MAX
}

/// Compares raw pointers for equality.
///
/// This is the same as using the `==` operator, but less generic:
/// the arguments have to be `*const T` raw pointers,
/// not anything that implements `PartialEq`.
///
/// This can be used to compare `&T` references (which coerce to `*const T` implicitly)
/// by their address rather than comparing the values they point to
/// (which is what the `PartialEq for &T` implementation does).
///
/// When comparing wide pointers, both the address and the metadata are tested for equality.
/// However, note that comparing trait object pointers (`*const dyn Trait`) is unreliable: pointers
/// to values of the same underlying type can compare inequal (because vtables are duplicated in
/// multiple codegen units), and pointers to values of *different* underlying type can compare equal
/// (since identical vtables can be deduplicated within a codegen unit).
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let five = 5;
/// let other_five = 5;
/// let five_ref = &five;
/// let same_five_ref = &five;
/// let other_five_ref = &other_five;
///
/// assert!(five_ref == same_five_ref);
/// assert!(ptr::eq(five_ref, same_five_ref));
///
/// assert!(five_ref == other_five_ref);
/// assert!(!ptr::eq(five_ref, other_five_ref));
/// ```
///
/// Slices are also compared by their length (fat pointers):
///
/// ```
/// let a = [1, 2, 3];
/// assert!(std::ptr::eq(&a[..3], &a[..3]));
/// assert!(!std::ptr::eq(&a[..2], &a[..3]));
/// assert!(!std::ptr::eq(&a[0..2], &a[1..3]));
/// ```
#[stable(feature = "ptr_eq", since = "1.17.0")]
#[inline(always)]
pub fn eq<T: ?Sized>(a: *const T, b: *const T) -> bool {
    a == b
}

/// Hash a raw pointer.
///
/// This can be used to hash a `&T` reference (which coerces to `*const T` implicitly)
/// by its address rather than the value it points to
/// (which is what the `Hash for &T` implementation does).
///
/// # Examples
///
/// ```
/// use std::collections::hash_map::DefaultHasher;
/// use std::hash::{Hash, Hasher};
/// use std::ptr;
///
/// let five = 5;
/// let five_ref = &five;
///
/// let mut hasher = DefaultHasher::new();
/// ptr::hash(five_ref, &mut hasher);
/// let actual = hasher.finish();
///
/// let mut hasher = DefaultHasher::new();
/// (five_ref as *const i32).hash(&mut hasher);
/// let expected = hasher.finish();
///
/// assert_eq!(actual, expected);
/// ```
#[stable(feature = "ptr_hash", since = "1.35.0")]
pub fn hash<T: ?Sized, S: hash::Hasher>(hashee: *const T, into: &mut S) {
    use crate::hash::Hash;
    hashee.hash(into);
}

// If this is a unary fn pointer, it adds a doc comment.
// Otherwise, it hides the docs entirely.
macro_rules! maybe_fnptr_doc {
    (@ #[$meta:meta] $item:item) => {
        #[doc(hidden)]
        #[$meta]
        $item
    };
    ($a:ident @ #[$meta:meta] $item:item) => {
        #[doc(fake_variadic)]
        #[doc = "This trait is implemented for function pointers with up to twelve arguments."]
        #[$meta]
        $item
    };
    ($a:ident $($rest_a:ident)+ @ #[$meta:meta] $item:item) => {
        #[doc(hidden)]
        #[$meta]
        $item
    };
}

// FIXME(strict_provenance_magic): function pointers have buggy codegen that
// necessitates casting to a usize to get the backend to do the right thing.
// for now I will break AVR to silence *a billion* lints. We should probably
// have a proper "opaque function pointer type" to handle this kind of thing.

// Impls for function pointers
macro_rules! fnptr_impls_safety_abi {
    ($FnTy: ty, $($Arg: ident),*) => {
        fnptr_impls_safety_abi! { #[stable(feature = "fnptr_impls", since = "1.4.0")] $FnTy, $($Arg),* }
    };
    (@c_unwind $FnTy: ty, $($Arg: ident),*) => {
        fnptr_impls_safety_abi! { #[unstable(feature = "c_unwind", issue = "74990")] $FnTy, $($Arg),* }
    };
    (#[$meta:meta] $FnTy: ty, $($Arg: ident),*) => {
        maybe_fnptr_doc! {
            $($Arg)* @
            #[$meta]
            impl<Ret, $($Arg),*> PartialEq for $FnTy {
                #[inline]
                fn eq(&self, other: &Self) -> bool {
                    *self as usize == *other as usize
                }
            }
        }

        maybe_fnptr_doc! {
            $($Arg)* @
            #[$meta]
            impl<Ret, $($Arg),*> Eq for $FnTy {}
        }

        maybe_fnptr_doc! {
            $($Arg)* @
            #[$meta]
            impl<Ret, $($Arg),*> PartialOrd for $FnTy {
                #[inline]
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    (*self as usize).partial_cmp(&(*other as usize))
                }
            }
        }

        maybe_fnptr_doc! {
            $($Arg)* @
            #[$meta]
            impl<Ret, $($Arg),*> Ord for $FnTy {
                #[inline]
                fn cmp(&self, other: &Self) -> Ordering {
                    (*self as usize).cmp(&(*other as usize))
                }
            }
        }

        maybe_fnptr_doc! {
            $($Arg)* @
            #[$meta]
            impl<Ret, $($Arg),*> hash::Hash for $FnTy {
                fn hash<HH: hash::Hasher>(&self, state: &mut HH) {
                    state.write_usize(*self as usize)
                }
            }
        }

        maybe_fnptr_doc! {
            $($Arg)* @
            #[$meta]
            impl<Ret, $($Arg),*> fmt::Pointer for $FnTy {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmt::pointer_fmt_inner(*self as usize, f)
                }
            }
        }

        maybe_fnptr_doc! {
            $($Arg)* @
            #[$meta]
            impl<Ret, $($Arg),*> fmt::Debug for $FnTy {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmt::pointer_fmt_inner(*self as usize, f)
                }
            }
        }
    }
}

macro_rules! fnptr_impls_args {
    ($($Arg: ident),+) => {
        fnptr_impls_safety_abi! { extern "Rust" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { extern "C" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { extern "C" fn($($Arg),+ , ...) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { @c_unwind extern "C-unwind" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { @c_unwind extern "C-unwind" fn($($Arg),+ , ...) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { unsafe extern "Rust" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { unsafe extern "C" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { unsafe extern "C" fn($($Arg),+ , ...) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { @c_unwind unsafe extern "C-unwind" fn($($Arg),+) -> Ret, $($Arg),+ }
        fnptr_impls_safety_abi! { @c_unwind unsafe extern "C-unwind" fn($($Arg),+ , ...) -> Ret, $($Arg),+ }
    };
    () => {
        // No variadic functions with 0 parameters
        fnptr_impls_safety_abi! { extern "Rust" fn() -> Ret, }
        fnptr_impls_safety_abi! { extern "C" fn() -> Ret, }
        fnptr_impls_safety_abi! { @c_unwind extern "C-unwind" fn() -> Ret, }
        fnptr_impls_safety_abi! { unsafe extern "Rust" fn() -> Ret, }
        fnptr_impls_safety_abi! { unsafe extern "C" fn() -> Ret, }
        fnptr_impls_safety_abi! { @c_unwind unsafe extern "C-unwind" fn() -> Ret, }
    };
}

fnptr_impls_args! {}
fnptr_impls_args! { T }
fnptr_impls_args! { A, B }
fnptr_impls_args! { A, B, C }
fnptr_impls_args! { A, B, C, D }
fnptr_impls_args! { A, B, C, D, E }
fnptr_impls_args! { A, B, C, D, E, F }
fnptr_impls_args! { A, B, C, D, E, F, G }
fnptr_impls_args! { A, B, C, D, E, F, G, H }
fnptr_impls_args! { A, B, C, D, E, F, G, H, I }
fnptr_impls_args! { A, B, C, D, E, F, G, H, I, J }
fnptr_impls_args! { A, B, C, D, E, F, G, H, I, J, K }
fnptr_impls_args! { A, B, C, D, E, F, G, H, I, J, K, L }

/// Create a `const` raw pointer to a place, without creating an intermediate reference.
///
/// Creating a reference with `&`/`&mut` is only allowed if the pointer is properly aligned
/// and points to initialized data. For cases where those requirements do not hold,
/// raw pointers should be used instead. However, `&expr as *const _` creates a reference
/// before casting it to a raw pointer, and that reference is subject to the same rules
/// as all other references. This macro can create a raw pointer *without* creating
/// a reference first.
///
/// Note, however, that the `expr` in `addr_of!(expr)` is still subject to all
/// the usual rules. In particular, `addr_of!(*ptr::null())` is Undefined
/// Behavior because it dereferences a null pointer.
///
/// # Example
///
/// ```
/// use std::ptr;
///
/// #[repr(packed)]
/// struct Packed {
///     f1: u8,
///     f2: u16,
/// }
///
/// let packed = Packed { f1: 1, f2: 2 };
/// // `&packed.f2` would create an unaligned reference, and thus be Undefined Behavior!
/// let raw_f2 = ptr::addr_of!(packed.f2);
/// assert_eq!(unsafe { raw_f2.read_unaligned() }, 2);
/// ```
///
/// See [`addr_of_mut`] for how to create a pointer to unininitialized data.
/// Doing that with `addr_of` would not make much sense since one could only
/// read the data, and that would be Undefined Behavior.
#[stable(feature = "raw_ref_macros", since = "1.51.0")]
#[rustc_macro_transparency = "semitransparent"]
#[allow_internal_unstable(raw_ref_op)]
pub macro addr_of($place:expr) {
    &raw const $place
}

/// Create a `mut` raw pointer to a place, without creating an intermediate reference.
///
/// Creating a reference with `&`/`&mut` is only allowed if the pointer is properly aligned
/// and points to initialized data. For cases where those requirements do not hold,
/// raw pointers should be used instead. However, `&mut expr as *mut _` creates a reference
/// before casting it to a raw pointer, and that reference is subject to the same rules
/// as all other references. This macro can create a raw pointer *without* creating
/// a reference first.
///
/// Note, however, that the `expr` in `addr_of_mut!(expr)` is still subject to all
/// the usual rules. In particular, `addr_of_mut!(*ptr::null_mut())` is Undefined
/// Behavior because it dereferences a null pointer.
///
/// # Examples
///
/// **Creating a pointer to unaligned data:**
///
/// ```
/// use std::ptr;
///
/// #[repr(packed)]
/// struct Packed {
///     f1: u8,
///     f2: u16,
/// }
///
/// let mut packed = Packed { f1: 1, f2: 2 };
/// // `&mut packed.f2` would create an unaligned reference, and thus be Undefined Behavior!
/// let raw_f2 = ptr::addr_of_mut!(packed.f2);
/// unsafe { raw_f2.write_unaligned(42); }
/// assert_eq!({packed.f2}, 42); // `{...}` forces copying the field instead of creating a reference.
/// ```
///
/// **Creating a pointer to uninitialized data:**
///
/// ```rust
/// use std::{ptr, mem::MaybeUninit};
///
/// struct Demo {
///     field: bool,
/// }
///
/// let mut uninit = MaybeUninit::<Demo>::uninit();
/// // `&uninit.as_mut().field` would create a reference to an uninitialized `bool`,
/// // and thus be Undefined Behavior!
/// let f1_ptr = unsafe { ptr::addr_of_mut!((*uninit.as_mut_ptr()).field) };
/// unsafe { f1_ptr.write(true); }
/// let init = unsafe { uninit.assume_init() };
/// ```
#[stable(feature = "raw_ref_macros", since = "1.51.0")]
#[rustc_macro_transparency = "semitransparent"]
#[allow_internal_unstable(raw_ref_op)]
pub macro addr_of_mut($place:expr) {
    &raw mut $place
}
