//! Manually manage memory through raw pointers.
//!
//! *[See also the pointer primitive types](pointer).*
//!
//! # Safety
//!
//! Many functions in this module take raw pointers as arguments and read from or write to them. For
//! this to be safe, these pointers must be *valid* for the given access. Whether a pointer is valid
//! depends on the operation it is used for (read or write), and the extent of the memory that is
//! accessed (i.e., how many bytes are read/written) -- it makes no sense to ask "is this pointer
//! valid"; one has to ask "is this pointer valid for a given access". Most functions use `*mut T`
//! and `*const T` to access only a single value, in which case the documentation omits the size and
//! implicitly assumes it to be `size_of::<T>()` bytes.
//!
//! The precise rules for validity are not determined yet. The guarantees that are
//! provided at this point are very minimal:
//!
//! * For memory accesses of [size zero][zst], *every* pointer is valid, including the [null]
//!   pointer. The following points are only concerned with non-zero-sized accesses.
//! * A [null] pointer is *never* valid.
//! * For a pointer to be valid, it is necessary, but not always sufficient, that the pointer be
//!   *dereferenceable*. The [provenance] of the pointer is used to determine which [allocated
//!   object] it is derived from; a pointer is dereferenceable if the memory range of the given size
//!   starting at the pointer is entirely contained within the bounds of that allocated object. Note
//!   that in Rust, every (stack-allocated) variable is considered a separate allocated object.
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
//! We say that a pointer is "dangling" if it is not valid for any non-zero-sized accesses. This
//! means out-of-bounds pointers, pointers to freed memory, null pointers, and pointers created with
//! [`NonNull::dangling`] are all dangling.
//!
//! ## Alignment
//!
//! Valid raw pointers as defined above are not necessarily properly aligned (where
//! "proper" alignment is defined by the pointee type, i.e., `*const T` must be
//! aligned to `align_of::<T>()`). However, most functions require their
//! arguments to be properly aligned, and will explicitly state
//! this requirement in their documentation. Notable exceptions to this are
//! [`read_unaligned`] and [`write_unaligned`].
//!
//! When a function requires proper alignment, it does so even if the access
//! has size 0, i.e., even if memory is not actually touched. Consider using
//! [`NonNull::dangling`] in such cases.
//!
//! ## Pointer to reference conversion
//!
//! When converting a pointer to a reference (e.g. via `&*ptr` or `&mut *ptr`),
//! there are several rules that must be followed:
//!
//! * The pointer must be properly aligned.
//!
//! * It must be non-null.
//!
//! * It must be "dereferenceable" in the sense defined above.
//!
//! * The pointer must point to a [valid value] of type `T`.
//!
//! * You must enforce Rust's aliasing rules. The exact aliasing rules are not decided yet, so we
//!   only give a rough overview here. The rules also depend on whether a mutable or a shared
//!   reference is being created.
//!   * When creating a mutable reference, then while this reference exists, the memory it points to
//!     must not get accessed (read or written) through any other pointer or reference not derived
//!     from this reference.
//!   * When creating a shared reference, then while this reference exists, the memory it points to
//!     must not get mutated (except inside `UnsafeCell`).
//!
//! If a pointer follows all of these rules, it is said to be
//! *convertible to a (mutable or shared) reference*.
// ^ we use this term instead of saying that the produced reference must
// be valid, as the validity of a reference is easily confused for the
// validity of the thing it refers to, and while the two concepts are
// closely related, they are not identical.
//!
//! These rules apply even if the result is unused!
//! (The part about being initialized is not yet fully decided, but until
//! it is, the only safe approach is to ensure that they are indeed initialized.)
//!
//! An example of the implications of the above rules is that an expression such
//! as `unsafe { &*(0 as *const u8) }` is Immediate Undefined Behavior.
//!
//! [valid value]: ../../reference/behavior-considered-undefined.html#invalid-values
//!
//! ## Allocated object
//!
//! An *allocated object* is a subset of program memory which is addressable
//! from Rust, and within which pointer arithmetic is possible. Examples of
//! allocated objects include heap allocations, stack-allocated variables,
//! statics, and consts. The safety preconditions of some Rust operations -
//! such as `offset` and field projections (`expr.field`) - are defined in
//! terms of the allocated objects on which they operate.
//!
//! An allocated object has a base address, a size, and a set of memory
//! addresses. It is possible for an allocated object to have zero size, but
//! such an allocated object will still have a base address. The base address
//! of an allocated object is not necessarily unique. While it is currently the
//! case that an allocated object always has a set of memory addresses which is
//! fully contiguous (i.e., has no "holes"), there is no guarantee that this
//! will not change in the future.
//!
//! For any allocated object with `base` address, `size`, and a set of
//! `addresses`, the following are guaranteed:
//! - For all addresses `a` in `addresses`, `a` is in the range `base .. (base +
//!   size)` (note that this requires `a < base + size`, not `a <= base + size`)
//! - `base` is not equal to [`null()`] (i.e., the address with the numerical
//!   value 0)
//! - `base + size <= usize::MAX`
//! - `size <= isize::MAX`
//!
//! As a consequence of these guarantees, given any address `a` within the set
//! of addresses of an allocated object:
//! - It is guaranteed that `a - base` does not overflow `isize`
//! - It is guaranteed that `a - base` is non-negative
//! - It is guaranteed that, given `o = a - base` (i.e., the offset of `a` within
//!   the allocated object), `base + o` will not wrap around the address space (in
//!   other words, will not overflow `usize`)
//!
//! [`null()`]: null
//!
//! # Provenance
//!
//! Pointers are not *simply* an "integer" or "address". For instance, it's uncontroversial
//! to say that a Use After Free is clearly Undefined Behavior, even if you "get lucky"
//! and the freed memory gets reallocated before your read/write (in fact this is the
//! worst-case scenario, UAFs would be much less concerning if this didn't happen!).
//! As another example, consider that [`wrapping_offset`] is documented to "remember"
//! the allocated object that the original pointer points to, even if it is offset far
//! outside the memory range occupied by that allocated object.
//! To rationalize claims like this, pointers need to somehow be *more* than just their addresses:
//! they must have **provenance**.
//!
//! A pointer value in Rust semantically contains the following information:
//!
//! * The **address** it points to, which can be represented by a `usize`.
//! * The **provenance** it has, defining the memory it has permission to access. Provenance can be
//!   absent, in which case the pointer does not have permission to access any memory.
//!
//! The exact structure of provenance is not yet specified, but the permission defined by a
//! pointer's provenance have a *spatial* component, a *temporal* component, and a *mutability*
//! component:
//!
//! * Spatial: The set of memory addresses that the pointer is allowed to access.
//! * Temporal: The timespan during which the pointer is allowed to access those memory addresses.
//! * Mutability: Whether the pointer may only access the memory for reads, or also access it for
//!   writes. Note that this can interact with the other components, e.g. a pointer might permit
//!   mutation only for a subset of addresses, or only for a subset of its maximal timespan.
//!
//! When an [allocated object] is created, it has a unique Original Pointer. For alloc
//! APIs this is literally the pointer the call returns, and for local variables and statics,
//! this is the name of the variable/static. (This is mildly overloading the term "pointer"
//! for the sake of brevity/exposition.)
//!
//! The Original Pointer for an allocated object has provenance that constrains the *spatial*
//! permissions of this pointer to the memory range of the allocation, and the *temporal*
//! permissions to the lifetime of the allocation. Provenance is implicitly inherited by all
//! pointers transitively derived from the Original Pointer through operations like [`offset`],
//! borrowing, and pointer casts. Some operations may *shrink* the permissions of the derived
//! provenance, limiting how much memory it can access or how long it's valid for (i.e. borrowing a
//! subfield and subslicing can shrink the spatial component of provenance, and all borrowing can
//! shrink the temporal component of provenance). However, no operation can ever *grow* the
//! permissions of the derived provenance: even if you "know" there is a larger allocation, you
//! can't derive a pointer with a larger provenance. Similarly, you cannot "recombine" two
//! contiguous provenances back into one (i.e. with a `fn merge(&[T], &[T]) -> &[T]`).
//!
//! A reference to a place always has provenance over at least the memory that place occupies.
//! A reference to a slice always has provenance over at least the range that slice describes.
//! Whether and when exactly the provenance of a reference gets "shrunk" to *exactly* fit
//! the memory it points to is not yet determined.
//!
//! A *shared* reference only ever has provenance that permits reading from memory,
//! and never permits writes, except inside [`UnsafeCell`].
//!
//! Provenance can affect whether a program has undefined behavior:
//!
//! * It is undefined behavior to access memory through a pointer that does not have provenance over
//!   that memory. Note that a pointer "at the end" of its provenance is not actually outside its
//!   provenance, it just has 0 bytes it can load/store. Zero-sized accesses do not require any
//!   provenance since they access an empty range of memory.
//!
//! * It is undefined behavior to [`offset`] a pointer across a memory range that is not contained
//!   in the allocated object it is derived from, or to [`offset_from`] two pointers not derived
//!   from the same allocated object. Provenance is used to say what exactly "derived from" even
//!   means: the lineage of a pointer is traced back to the Original Pointer it descends from, and
//!   that identifies the relevant allocated object. In particular, it's always UB to offset a
//!   pointer derived from something that is now deallocated, except if the offset is 0.
//!
//! But it *is* still sound to:
//!
//! * Create a pointer without provenance from just an address (see [`without_provenance`]). Such a
//!   pointer cannot be used for memory accesses (except for zero-sized accesses). This can still be
//!   useful for sentinel values like `null` *or* to represent a tagged pointer that will never be
//!   dereferenceable. In general, it is always sound for an integer to pretend to be a pointer "for
//!   fun" as long as you don't use operations on it which require it to be valid (non-zero-sized
//!   offset, read, write, etc).
//!
//! * Forge an allocation of size zero at any sufficiently aligned non-null address.
//!   i.e. the usual "ZSTs are fake, do what you want" rules apply.
//!
//! * [`wrapping_offset`] a pointer outside its provenance. This includes pointers
//!   which have "no" provenance. In particular, this makes it sound to do pointer tagging tricks.
//!
//! * Compare arbitrary pointers by address. Pointer comparison ignores provenance and addresses
//!   *are* just integers, so there is always a coherent answer, even if the pointers are dangling
//!   or from different provenances. Note that if you get "lucky" and notice that a pointer at the
//!   end of one allocated object is the "same" address as the start of another allocated object,
//!   anything you do with that fact is *probably* going to be gibberish. The scope of that
//!   gibberish is kept under control by the fact that the two pointers *still* aren't allowed to
//!   access the other's allocation (bytes), because they still have different provenance.
//!
//! Note that the full definition of provenance in Rust is not decided yet, as this interacts
//! with the as-yet undecided [aliasing] rules.
//!
//! ## Pointers Vs Integers
//!
//! From this discussion, it becomes very clear that a `usize` *cannot* accurately represent a pointer,
//! and converting from a pointer to a `usize` is generally an operation which *only* extracts the
//! address. Converting this address back into pointer requires somehow answering the question:
//! which provenance should the resulting pointer have?
//!
//! Rust provides two ways of dealing with this situation: *Strict Provenance* and *Exposed Provenance*.
//!
//! Note that a pointer *can* represent a `usize` (via [`without_provenance`]), so the right type to
//! use in situations where a value is "sometimes a pointer and sometimes a bare `usize`" is a
//! pointer type.
//!
//! ## Strict Provenance
//!
//! "Strict Provenance" refers to a set of APIs designed to make working with provenance more
//! explicit. They are intended as substitutes for casting a pointer to an integer and back.
//!
//! Entirely avoiding integer-to-pointer casts successfully side-steps the inherent ambiguity of
//! that operation. This benefits compiler optimizations, and it is pretty much a requirement for
//! using tools like [Miri] and architectures like [CHERI] that aim to detect and diagnose pointer
//! misuse.
//!
//! The key insight to making programming without integer-to-pointer casts *at all* viable is the
//! [`with_addr`] method:
//!
//! ```text
//!     /// Creates a new pointer with the given address.
//!     ///
//!     /// This performs the same operation as an `addr as ptr` cast, but copies
//!     /// the *provenance* of `self` to the new pointer.
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
//! into the allocation you care about that can "reconstitute" the provenance.
//! Usually this is very easy, because you only are taking a pointer, messing with the address,
//! and then immediately converting back to a pointer. To make this use case more ergonomic,
//! we provide the [`map_addr`] method.
//!
//! To help make it clear that code is "following" Strict Provenance semantics, we also provide an
//! [`addr`] method which promises that the returned address is not part of a
//! pointer-integer-pointer roundtrip. In the future we may provide a lint for pointer<->integer
//! casts to help you audit if your code conforms to strict provenance.
//!
//! ### Using Strict Provenance
//!
//! Most code needs no changes to conform to strict provenance, as the only really concerning
//! operation is casts from `usize` to a pointer. For code which *does* cast a `usize` to a pointer,
//! the scope of the change depends on exactly what you're doing.
//!
//! In general, you just need to make sure that if you want to convert a `usize` address to a
//! pointer and then use that pointer to read/write memory, you need to keep around a pointer
//! that has sufficient provenance to perform that read/write itself. In this way all of your
//! casts from an address to a pointer are essentially just applying offsets/indexing.
//!
//! This is generally trivial to do for simple cases like tagged pointers *as long as you
//! represent the tagged pointer as an actual pointer and not a `usize`*. For instance:
//!
//! ```
//! unsafe {
//!     // A flag we want to pack into our pointer
//!     static HAS_DATA: usize = 0x1;
//!     static FLAG_MASK: usize = !HAS_DATA;
//!
//!     // Our value, which must have enough alignment to have spare least-significant-bits.
//!     let my_precious_data: u32 = 17;
//!     assert!(align_of::<u32>() > 1);
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
//! (Yes, if you've been using [`AtomicUsize`] for pointers in concurrent datastructures, you should
//! be using [`AtomicPtr`] instead. If that messes up the way you atomically manipulate pointers,
//! we would like to know why, and what needs to be done to fix it.)
//!
//! Situations where a valid pointer *must* be created from just an address, such as baremetal code
//! accessing a memory-mapped interface at a fixed address, cannot currently be handled with strict
//! provenance APIs and should use [exposed provenance](#exposed-provenance).
//!
//! ## Exposed Provenance
//!
//! As discussed above, integer-to-pointer casts are not possible with Strict Provenance APIs.
//! This is by design: the goal of Strict Provenance is to provide a clear specification that we are
//! confident can be formalized unambiguously and can be subject to precise formal reasoning.
//! Integer-to-pointer casts do not (currently) have such a clear specification.
//!
//! However, there exist situations where integer-to-pointer casts cannot be avoided, or
//! where avoiding them would require major refactoring. Legacy platform APIs also regularly assume
//! that `usize` can capture all the information that makes up a pointer.
//! Bare-metal platforms can also require the synthesis of a pointer "out of thin air" without
//! anywhere to obtain proper provenance from.
//!
//! Rust's model for dealing with integer-to-pointer casts is called *Exposed Provenance*. However,
//! the semantics of Exposed Provenance are on much less solid footing than Strict Provenance, and
//! at this point it is not yet clear whether a satisfying unambiguous semantics can be defined for
//! Exposed Provenance. (If that sounds bad, be reassured that other popular languages that provide
//! integer-to-pointer casts are not faring any better.) Furthermore, Exposed Provenance will not
//! work (well) with tools like [Miri] and [CHERI].
//!
//! Exposed Provenance is provided by the [`expose_provenance`] and [`with_exposed_provenance`] methods,
//! which are equivalent to `as` casts between pointers and integers.
//! - [`expose_provenance`] is a lot like [`addr`], but additionally adds the provenance of the
//!   pointer to a global list of 'exposed' provenances. (This list is purely conceptual, it exists
//!   for the purpose of specifying Rust but is not materialized in actual executions, except in
//!   tools like [Miri].)
//!   Memory which is outside the control of the Rust abstract machine (MMIO registers, for example)
//!   is always considered to be exposed, so long as this memory is disjoint from memory that will
//!   be used by the abstract machine such as the stack, heap, and statics.
//! - [`with_exposed_provenance`] can be used to construct a pointer with one of these previously
//!   'exposed' provenances. [`with_exposed_provenance`] takes only `addr: usize` as arguments, so
//!   unlike in [`with_addr`] there is no indication of what the correct provenance for the returned
//!   pointer is -- and that is exactly what makes integer-to-pointer casts so tricky to rigorously
//!   specify! The compiler will do its best to pick the right provenance for you, but currently we
//!   cannot provide any guarantees about which provenance the resulting pointer will have. Only one
//!   thing is clear: if there is *no* previously 'exposed' provenance that justifies the way the
//!   returned pointer will be used, the program has undefined behavior.
//!
//! If at all possible, we encourage code to be ported to [Strict Provenance] APIs, thus avoiding
//! the need for Exposed Provenance. Maximizing the amount of such code is a major win for avoiding
//! specification complexity and to facilitate adoption of tools like [CHERI] and [Miri] that can be
//! a big help in increasing the confidence in (unsafe) Rust code. However, we acknowledge that this
//! is not always possible, and offer Exposed Provenance as a way to explicit "opt out" of the
//! well-defined semantics of Strict Provenance, and "opt in" to the unclear semantics of
//! integer-to-pointer casts.
//!
//! [aliasing]: ../../nomicon/aliasing.html
//! [allocated object]: #allocated-object
//! [provenance]: #provenance
//! [book]: ../../book/ch19-01-unsafe-rust.html#dereferencing-a-raw-pointer
//! [ub]: ../../reference/behavior-considered-undefined.html
//! [zst]: ../../nomicon/exotic-sizes.html#zero-sized-types-zsts
//! [atomic operations]: crate::sync::atomic
//! [`offset`]: pointer::offset
//! [`offset_from`]: pointer::offset_from
//! [`wrapping_offset`]: pointer::wrapping_offset
//! [`with_addr`]: pointer::with_addr
//! [`map_addr`]: pointer::map_addr
//! [`addr`]: pointer::addr
//! [`AtomicUsize`]: crate::sync::atomic::AtomicUsize
//! [`AtomicPtr`]: crate::sync::atomic::AtomicPtr
//! [`expose_provenance`]: pointer::expose_provenance
//! [`with_exposed_provenance`]: with_exposed_provenance
//! [Miri]: https://github.com/rust-lang/miri
//! [CHERI]: https://www.cl.cam.ac.uk/research/security/ctsrd/cheri/
//! [Strict Provenance]: #strict-provenance
//! [`UnsafeCell`]: core::cell::UnsafeCell

#![stable(feature = "rust1", since = "1.0.0")]
// There are many unsafe functions taking pointers that don't dereference them.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::cmp::Ordering;
use crate::intrinsics::const_eval_select;
use crate::marker::FnPtr;
use crate::mem::{self, MaybeUninit, SizedTypeProperties};
use crate::num::NonZero;
use crate::{fmt, hash, intrinsics, ub_checks};

mod alignment;
#[unstable(feature = "ptr_alignment_type", issue = "102070")]
pub use alignment::Alignment;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::copy;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::copy_nonoverlapping;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use crate::intrinsics::write_bytes;

mod metadata;
#[unstable(feature = "ptr_metadata", issue = "81513")]
pub use metadata::{DynMetadata, Pointee, Thin, from_raw_parts, from_raw_parts_mut, metadata};

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
/// This is almost the same as calling [`ptr::read`] and discarding
/// the result, but has the following advantages:
// FIXME: say something more useful than "almost the same"?
// There are open questions here: `read` requires the value to be fully valid, e.g. if `T` is a
// `bool` it must be 0 or 1, if it is a reference then it must be dereferenceable. `drop_in_place`
// only requires that `*to_drop` be "valid for dropping" and we have not defined what that means. In
// Miri it currently (May 2024) requires nothing at all for types without drop glue.
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
/// * `to_drop` must be properly aligned, even if `T` has size 0.
///
/// * `to_drop` must be nonnull, even if `T` has size 0.
///
/// * The value `to_drop` points to must be valid for dropping, which may mean
///   it must uphold additional invariants. These invariants depend on the type
///   of the value being dropped. For instance, when dropping a Box, the box's
///   pointer to the heap must be valid.
///
/// * While `drop_in_place` is executing, the only way to access parts of
///   `to_drop` is through the `&mut self` references supplied to the
///   `Drop::drop` methods that `drop_in_place` invokes.
///
/// Additionally, if `T` is not [`Copy`], using the pointed-to value after
/// calling `drop_in_place` can cause undefined behavior. Note that `*to_drop =
/// foo` counts as a use because it will cause the value to be dropped
/// again. [`write()`] can be used to overwrite data without causing it to be
/// dropped.
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
#[rustc_diagnostic_item = "ptr_drop_in_place"]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.

    // SAFETY: see comment above
    unsafe { drop_in_place(to_drop) }
}

/// Creates a null raw pointer.
///
/// This function is equivalent to zero-initializing the pointer:
/// `MaybeUninit::<*const T>::zeroed().assume_init()`.
/// The resulting pointer has the address 0.
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let p: *const i32 = ptr::null();
/// assert!(p.is_null());
/// assert_eq!(p as usize, 0); // this pointer has the address 0
/// ```
#[inline(always)]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
#[rustc_const_stable(feature = "const_ptr_null", since = "1.24.0")]
#[rustc_diagnostic_item = "ptr_null"]
pub const fn null<T: ?Sized + Thin>() -> *const T {
    from_raw_parts(without_provenance::<()>(0), ())
}

/// Creates a null mutable raw pointer.
///
/// This function is equivalent to zero-initializing the pointer:
/// `MaybeUninit::<*mut T>::zeroed().assume_init()`.
/// The resulting pointer has the address 0.
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let p: *mut i32 = ptr::null_mut();
/// assert!(p.is_null());
/// assert_eq!(p as usize, 0); // this pointer has the address 0
/// ```
#[inline(always)]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_promotable]
#[rustc_const_stable(feature = "const_ptr_null", since = "1.24.0")]
#[rustc_diagnostic_item = "ptr_null_mut"]
pub const fn null_mut<T: ?Sized + Thin>() -> *mut T {
    from_raw_parts_mut(without_provenance_mut::<()>(0), ())
}

/// Creates a pointer with the given address and no [provenance][crate::ptr#provenance].
///
/// This is equivalent to `ptr::null().with_addr(addr)`.
///
/// Without provenance, this pointer is not associated with any actual allocation. Such a
/// no-provenance pointer may be used for zero-sized memory accesses (if suitably aligned), but
/// non-zero-sized memory accesses with a no-provenance pointer are UB. No-provenance pointers are
/// little more than a `usize` address in disguise.
///
/// This is different from `addr as *const T`, which creates a pointer that picks up a previously
/// exposed provenance. See [`with_exposed_provenance`] for more details on that operation.
///
/// This is a [Strict Provenance][crate::ptr#strict-provenance] API.
#[inline(always)]
#[must_use]
#[stable(feature = "strict_provenance", since = "1.84.0")]
#[rustc_const_stable(feature = "strict_provenance", since = "1.84.0")]
pub const fn without_provenance<T>(addr: usize) -> *const T {
    without_provenance_mut(addr)
}

/// Creates a new pointer that is dangling, but non-null and well-aligned.
///
/// This is useful for initializing types which lazily allocate, like
/// `Vec::new` does.
///
/// Note that the pointer value may potentially represent a valid pointer to
/// a `T`, which means this must not be used as a "not yet initialized"
/// sentinel value. Types that lazily allocate must track initialization by
/// some other means.
#[inline(always)]
#[must_use]
#[stable(feature = "strict_provenance", since = "1.84.0")]
#[rustc_const_stable(feature = "strict_provenance", since = "1.84.0")]
pub const fn dangling<T>() -> *const T {
    dangling_mut()
}

/// Creates a pointer with the given address and no [provenance][crate::ptr#provenance].
///
/// This is equivalent to `ptr::null_mut().with_addr(addr)`.
///
/// Without provenance, this pointer is not associated with any actual allocation. Such a
/// no-provenance pointer may be used for zero-sized memory accesses (if suitably aligned), but
/// non-zero-sized memory accesses with a no-provenance pointer are UB. No-provenance pointers are
/// little more than a `usize` address in disguise.
///
/// This is different from `addr as *mut T`, which creates a pointer that picks up a previously
/// exposed provenance. See [`with_exposed_provenance_mut`] for more details on that operation.
///
/// This is a [Strict Provenance][crate::ptr#strict-provenance] API.
#[inline(always)]
#[must_use]
#[stable(feature = "strict_provenance", since = "1.84.0")]
#[rustc_const_stable(feature = "strict_provenance", since = "1.84.0")]
pub const fn without_provenance_mut<T>(addr: usize) -> *mut T {
    // An int-to-pointer transmute currently has exactly the intended semantics: it creates a
    // pointer without provenance. Note that this is *not* a stable guarantee about transmute
    // semantics, it relies on sysroot crates having special status.
    // SAFETY: every valid integer is also a valid pointer (as long as you don't dereference that
    // pointer).
    unsafe { mem::transmute(addr) }
}

/// Creates a new pointer that is dangling, but non-null and well-aligned.
///
/// This is useful for initializing types which lazily allocate, like
/// `Vec::new` does.
///
/// Note that the pointer value may potentially represent a valid pointer to
/// a `T`, which means this must not be used as a "not yet initialized"
/// sentinel value. Types that lazily allocate must track initialization by
/// some other means.
#[inline(always)]
#[must_use]
#[stable(feature = "strict_provenance", since = "1.84.0")]
#[rustc_const_stable(feature = "strict_provenance", since = "1.84.0")]
pub const fn dangling_mut<T>() -> *mut T {
    NonNull::dangling().as_ptr()
}

/// Converts an address back to a pointer, picking up some previously 'exposed'
/// [provenance][crate::ptr#provenance].
///
/// This is fully equivalent to `addr as *const T`. The provenance of the returned pointer is that
/// of *some* pointer that was previously exposed by passing it to
/// [`expose_provenance`][pointer::expose_provenance], or a `ptr as usize` cast. In addition, memory
/// which is outside the control of the Rust abstract machine (MMIO registers, for example) is
/// always considered to be accessible with an exposed provenance, so long as this memory is disjoint
/// from memory that will be used by the abstract machine such as the stack, heap, and statics.
///
/// The exact provenance that gets picked is not specified. The compiler will do its best to pick
/// the "right" provenance for you (whatever that may be), but currently we cannot provide any
/// guarantees about which provenance the resulting pointer will have -- and therefore there
/// is no definite specification for which memory the resulting pointer may access.
///
/// If there is *no* previously 'exposed' provenance that justifies the way the returned pointer
/// will be used, the program has undefined behavior. In particular, the aliasing rules still apply:
/// pointers and references that have been invalidated due to aliasing accesses cannot be used
/// anymore, even if they have been exposed!
///
/// Due to its inherent ambiguity, this operation may not be supported by tools that help you to
/// stay conformant with the Rust memory model. It is recommended to use [Strict
/// Provenance][self#strict-provenance] APIs such as [`with_addr`][pointer::with_addr] wherever
/// possible.
///
/// On most platforms this will produce a value with the same bytes as the address. Platforms
/// which need to store additional information in a pointer may not support this operation,
/// since it is generally not possible to actually *compute* which provenance the returned
/// pointer has to pick up.
///
/// This is an [Exposed Provenance][crate::ptr#exposed-provenance] API.
#[must_use]
#[inline(always)]
#[stable(feature = "exposed_provenance", since = "1.84.0")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
#[allow(fuzzy_provenance_casts)] // this *is* the explicit provenance API one should use instead
pub fn with_exposed_provenance<T>(addr: usize) -> *const T {
    addr as *const T
}

/// Converts an address back to a mutable pointer, picking up some previously 'exposed'
/// [provenance][crate::ptr#provenance].
///
/// This is fully equivalent to `addr as *mut T`. The provenance of the returned pointer is that
/// of *some* pointer that was previously exposed by passing it to
/// [`expose_provenance`][pointer::expose_provenance], or a `ptr as usize` cast. In addition, memory
/// which is outside the control of the Rust abstract machine (MMIO registers, for example) is
/// always considered to be accessible with an exposed provenance, so long as this memory is disjoint
/// from memory that will be used by the abstract machine such as the stack, heap, and statics.
///
/// The exact provenance that gets picked is not specified. The compiler will do its best to pick
/// the "right" provenance for you (whatever that may be), but currently we cannot provide any
/// guarantees about which provenance the resulting pointer will have -- and therefore there
/// is no definite specification for which memory the resulting pointer may access.
///
/// If there is *no* previously 'exposed' provenance that justifies the way the returned pointer
/// will be used, the program has undefined behavior. In particular, the aliasing rules still apply:
/// pointers and references that have been invalidated due to aliasing accesses cannot be used
/// anymore, even if they have been exposed!
///
/// Due to its inherent ambiguity, this operation may not be supported by tools that help you to
/// stay conformant with the Rust memory model. It is recommended to use [Strict
/// Provenance][self#strict-provenance] APIs such as [`with_addr`][pointer::with_addr] wherever
/// possible.
///
/// On most platforms this will produce a value with the same bytes as the address. Platforms
/// which need to store additional information in a pointer may not support this operation,
/// since it is generally not possible to actually *compute* which provenance the returned
/// pointer has to pick up.
///
/// This is an [Exposed Provenance][crate::ptr#exposed-provenance] API.
#[must_use]
#[inline(always)]
#[stable(feature = "exposed_provenance", since = "1.84.0")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
#[allow(fuzzy_provenance_casts)] // this *is* the explicit provenance API one should use instead
pub fn with_exposed_provenance_mut<T>(addr: usize) -> *mut T {
    addr as *mut T
}

/// Converts a reference to a raw pointer.
///
/// For `r: &T`, `from_ref(r)` is equivalent to `r as *const T` (except for the caveat noted below),
/// but is a bit safer since it will never silently change type or mutability, in particular if the
/// code is refactored.
///
/// The caller must ensure that the pointee outlives the pointer this function returns, or else it
/// will end up dangling.
///
/// The caller must also ensure that the memory the pointer (non-transitively) points to is never
/// written to (except inside an `UnsafeCell`) using this pointer or any pointer derived from it. If
/// you need to mutate the pointee, use [`from_mut`]. Specifically, to turn a mutable reference `m:
/// &mut T` into `*const T`, prefer `from_mut(m).cast_const()` to obtain a pointer that can later be
/// used for mutation.
///
/// ## Interaction with lifetime extension
///
/// Note that this has subtle interactions with the rules for lifetime extension of temporaries in
/// tail expressions. This code is valid, albeit in a non-obvious way:
/// ```rust
/// # type T = i32;
/// # fn foo() -> T { 42 }
/// // The temporary holding the return value of `foo` has its lifetime extended,
/// // because the surrounding expression involves no function call.
/// let p = &foo() as *const T;
/// unsafe { p.read() };
/// ```
/// Naively replacing the cast with `from_ref` is not valid:
/// ```rust,no_run
/// # use std::ptr;
/// # type T = i32;
/// # fn foo() -> T { 42 }
/// // The temporary holding the return value of `foo` does *not* have its lifetime extended,
/// // because the surrounding expression involves a function call.
/// let p = ptr::from_ref(&foo());
/// unsafe { p.read() }; // UB! Reading from a dangling pointer ⚠️
/// ```
/// The recommended way to write this code is to avoid relying on lifetime extension
/// when raw pointers are involved:
/// ```rust
/// # use std::ptr;
/// # type T = i32;
/// # fn foo() -> T { 42 }
/// let x = foo();
/// let p = ptr::from_ref(&x);
/// unsafe { p.read() };
/// ```
#[inline(always)]
#[must_use]
#[stable(feature = "ptr_from_ref", since = "1.76.0")]
#[rustc_const_stable(feature = "ptr_from_ref", since = "1.76.0")]
#[rustc_never_returns_null_ptr]
#[rustc_diagnostic_item = "ptr_from_ref"]
pub const fn from_ref<T: ?Sized>(r: &T) -> *const T {
    r
}

/// Converts a mutable reference to a raw pointer.
///
/// For `r: &mut T`, `from_mut(r)` is equivalent to `r as *mut T` (except for the caveat noted
/// below), but is a bit safer since it will never silently change type or mutability, in particular
/// if the code is refactored.
///
/// The caller must ensure that the pointee outlives the pointer this function returns, or else it
/// will end up dangling.
///
/// ## Interaction with lifetime extension
///
/// Note that this has subtle interactions with the rules for lifetime extension of temporaries in
/// tail expressions. This code is valid, albeit in a non-obvious way:
/// ```rust
/// # type T = i32;
/// # fn foo() -> T { 42 }
/// // The temporary holding the return value of `foo` has its lifetime extended,
/// // because the surrounding expression involves no function call.
/// let p = &mut foo() as *mut T;
/// unsafe { p.write(T::default()) };
/// ```
/// Naively replacing the cast with `from_mut` is not valid:
/// ```rust,no_run
/// # use std::ptr;
/// # type T = i32;
/// # fn foo() -> T { 42 }
/// // The temporary holding the return value of `foo` does *not* have its lifetime extended,
/// // because the surrounding expression involves a function call.
/// let p = ptr::from_mut(&mut foo());
/// unsafe { p.write(T::default()) }; // UB! Writing to a dangling pointer ⚠️
/// ```
/// The recommended way to write this code is to avoid relying on lifetime extension
/// when raw pointers are involved:
/// ```rust
/// # use std::ptr;
/// # type T = i32;
/// # fn foo() -> T { 42 }
/// let mut x = foo();
/// let p = ptr::from_mut(&mut x);
/// unsafe { p.write(T::default()) };
/// ```
#[inline(always)]
#[must_use]
#[stable(feature = "ptr_from_ref", since = "1.76.0")]
#[rustc_const_stable(feature = "ptr_from_ref", since = "1.76.0")]
#[rustc_never_returns_null_ptr]
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
///
/// You must ensure that the pointer is valid and not null before dereferencing
/// the raw slice. A slice reference must never have a null pointer, even if it's empty.
///
/// ```rust,should_panic
/// use std::ptr;
/// let danger: *const [u8] = ptr::slice_from_raw_parts(ptr::null(), 0);
/// unsafe {
///     danger.as_ref().expect("references must not be null");
/// }
/// ```
#[inline]
#[stable(feature = "slice_from_raw_parts", since = "1.42.0")]
#[rustc_const_stable(feature = "const_slice_from_raw_parts", since = "1.64.0")]
#[rustc_diagnostic_item = "ptr_slice_from_raw_parts"]
pub const fn slice_from_raw_parts<T>(data: *const T, len: usize) -> *const [T] {
    from_raw_parts(data, len)
}

/// Forms a raw mutable slice from a pointer and a length.
///
/// The `len` argument is the number of **elements**, not the number of bytes.
///
/// Performs the same functionality as [`slice_from_raw_parts`], except that a
/// raw mutable slice is returned, as opposed to a raw immutable slice.
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
///
/// You must ensure that the pointer is valid and not null before dereferencing
/// the raw slice. A slice reference must never have a null pointer, even if it's empty.
///
/// ```rust,should_panic
/// use std::ptr;
/// let danger: *mut [u8] = ptr::slice_from_raw_parts_mut(ptr::null_mut(), 0);
/// unsafe {
///     danger.as_mut().expect("references must not be null");
/// }
/// ```
#[inline]
#[stable(feature = "slice_from_raw_parts", since = "1.42.0")]
#[rustc_const_stable(feature = "const_slice_from_raw_parts_mut", since = "1.83.0")]
#[rustc_diagnostic_item = "ptr_slice_from_raw_parts_mut"]
pub const fn slice_from_raw_parts_mut<T>(data: *mut T, len: usize) -> *mut [T] {
    from_raw_parts_mut(data, len)
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
/// * Both `x` and `y` must be [valid] for both reads and writes. They must remain valid even when the
///   other pointer is written. (This means if the memory ranges overlap, the two pointers must not
///   be subject to aliasing restrictions relative to each other.)
///
/// * Both `x` and `y` must be properly aligned.
///
/// Note that even if `T` has size `0`, the pointers must be properly aligned.
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
#[rustc_const_stable(feature = "const_swap", since = "1.85.0")]
#[rustc_diagnostic_item = "ptr_swap"]
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
/// the pointers must be properly aligned.
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
#[rustc_const_unstable(feature = "const_swap_nonoverlapping", issue = "133668")]
#[rustc_diagnostic_item = "ptr_swap_nonoverlapping"]
pub const unsafe fn swap_nonoverlapping<T>(x: *mut T, y: *mut T, count: usize) {
    ub_checks::assert_unsafe_precondition!(
        check_library_ub,
        "ptr::swap_nonoverlapping requires that both pointer arguments are aligned and non-null \
        and the specified memory ranges do not overlap",
        (
            x: *mut () = x as *mut (),
            y: *mut () = y as *mut (),
            size: usize = size_of::<T>(),
            align: usize = align_of::<T>(),
            count: usize = count,
        ) => {
            let zero_size = size == 0 || count == 0;
            ub_checks::maybe_is_aligned_and_not_null(x, align, zero_size)
                && ub_checks::maybe_is_aligned_and_not_null(y, align, zero_size)
                && ub_checks::maybe_is_nonoverlapping(x, y, size, count)
        }
    );

    const_eval_select!(
        @capture[T] { x: *mut T, y: *mut T, count: usize }:
        if const {
            // At compile-time we want to always copy this in chunks of `T`, to ensure that if there
            // are pointers inside `T` we will copy them in one go rather than trying to copy a part
            // of a pointer (which would not work).
            // SAFETY: Same preconditions as this function
            unsafe { swap_nonoverlapping_const(x, y, count) }
        } else {
            // Going though a slice here helps codegen know the size fits in `isize`
            let slice = slice_from_raw_parts_mut(x, count);
            // SAFETY: This is all readable from the pointer, meaning it's one
            // allocated object, and thus cannot be more than isize::MAX bytes.
            let bytes = unsafe { mem::size_of_val_raw::<[T]>(slice) };
            if let Some(bytes) = NonZero::new(bytes) {
                // SAFETY: These are the same ranges, just expressed in a different
                // type, so they're still non-overlapping.
                unsafe { swap_nonoverlapping_bytes(x.cast(), y.cast(), bytes) };
            }
        }
    )
}

/// Same behavior and safety conditions as [`swap_nonoverlapping`]
#[inline]
const unsafe fn swap_nonoverlapping_const<T>(x: *mut T, y: *mut T, count: usize) {
    let mut i = 0;
    while i < count {
        // SAFETY: By precondition, `i` is in-bounds because it's below `n`
        let x = unsafe { x.add(i) };
        // SAFETY: By precondition, `i` is in-bounds because it's below `n`
        // and it's distinct from `x` since the ranges are non-overlapping
        let y = unsafe { y.add(i) };

        // SAFETY: we're only ever given pointers that are valid to read/write,
        // including being aligned, and nothing here panics so it's drop-safe.
        unsafe {
            // Note that it's critical that these use `copy_nonoverlapping`,
            // rather than `read`/`write`, to avoid #134713 if T has padding.
            let mut temp = MaybeUninit::<T>::uninit();
            copy_nonoverlapping(x, temp.as_mut_ptr(), 1);
            copy_nonoverlapping(y, x, 1);
            copy_nonoverlapping(temp.as_ptr(), y, 1);
        }

        i += 1;
    }
}

// Don't let MIR inline this, because we really want it to keep its noalias metadata
#[rustc_no_mir_inline]
#[inline]
fn swap_chunk<const N: usize>(x: &mut MaybeUninit<[u8; N]>, y: &mut MaybeUninit<[u8; N]>) {
    let a = *x;
    let b = *y;
    *x = b;
    *y = a;
}

#[inline]
unsafe fn swap_nonoverlapping_bytes(x: *mut u8, y: *mut u8, bytes: NonZero<usize>) {
    // Same as `swap_nonoverlapping::<[u8; N]>`.
    unsafe fn swap_nonoverlapping_chunks<const N: usize>(
        x: *mut MaybeUninit<[u8; N]>,
        y: *mut MaybeUninit<[u8; N]>,
        chunks: NonZero<usize>,
    ) {
        let chunks = chunks.get();
        for i in 0..chunks {
            // SAFETY: i is in [0, chunks) so the adds and dereferences are in-bounds.
            unsafe { swap_chunk(&mut *x.add(i), &mut *y.add(i)) };
        }
    }

    // Same as `swap_nonoverlapping_bytes`, but accepts at most 1+2+4=7 bytes
    #[inline]
    unsafe fn swap_nonoverlapping_short(x: *mut u8, y: *mut u8, bytes: NonZero<usize>) {
        // Tail handling for auto-vectorized code sometimes has element-at-a-time behaviour,
        // see <https://github.com/rust-lang/rust/issues/134946>.
        // By swapping as different sizes, rather than as a loop over bytes,
        // we make sure not to end up with, say, seven byte-at-a-time copies.

        let bytes = bytes.get();
        let mut i = 0;
        macro_rules! swap_prefix {
            ($($n:literal)+) => {$(
                if (bytes & $n) != 0 {
                    // SAFETY: `i` can only have the same bits set as those in bytes,
                    // so these `add`s are in-bounds of `bytes`.  But the bit for
                    // `$n` hasn't been set yet, so the `$n` bytes that `swap_chunk`
                    // will read and write are within the usable range.
                    unsafe { swap_chunk::<$n>(&mut*x.add(i).cast(), &mut*y.add(i).cast()) };
                    i |= $n;
                }
            )+};
        }
        swap_prefix!(4 2 1);
        debug_assert_eq!(i, bytes);
    }

    const CHUNK_SIZE: usize = size_of::<*const ()>();
    let bytes = bytes.get();

    let chunks = bytes / CHUNK_SIZE;
    let tail = bytes % CHUNK_SIZE;
    if let Some(chunks) = NonZero::new(chunks) {
        // SAFETY: this is bytes/CHUNK_SIZE*CHUNK_SIZE bytes, which is <= bytes,
        // so it's within the range of our non-overlapping bytes.
        unsafe { swap_nonoverlapping_chunks::<CHUNK_SIZE>(x.cast(), y.cast(), chunks) };
    }
    if let Some(tail) = NonZero::new(tail) {
        const { assert!(CHUNK_SIZE <= 8) };
        let delta = chunks * CHUNK_SIZE;
        // SAFETY: the tail length is below CHUNK SIZE because of the remainder,
        // and CHUNK_SIZE is at most 8 by the const assert, so tail <= 7
        unsafe { swap_nonoverlapping_short(x.add(delta), y.add(delta), tail) };
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
/// Note that even if `T` has size `0`, the pointer must be properly aligned.
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
#[rustc_const_stable(feature = "const_replace", since = "1.83.0")]
#[rustc_diagnostic_item = "ptr_replace"]
pub const unsafe fn replace<T>(dst: *mut T, src: T) -> T {
    // SAFETY: the caller must guarantee that `dst` is valid to be
    // cast to a mutable reference (valid for writes, aligned, initialized),
    // and cannot overlap `src` since `dst` must point to a distinct
    // allocated object.
    unsafe {
        ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "ptr::replace requires that the pointer argument is aligned and non-null",
            (
                addr: *const () = dst as *const (),
                align: usize = align_of::<T>(),
                is_zst: bool = T::IS_ZST,
            ) => ub_checks::maybe_is_aligned_and_not_null(addr, align, is_zst)
        );
        mem::replace(&mut *dst, src)
    }
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
/// Note that even if `T` has size `0`, the pointer must be properly aligned.
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
#[rustc_const_stable(feature = "const_ptr_read", since = "1.71.0")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
#[rustc_diagnostic_item = "ptr_read"]
pub const unsafe fn read<T>(src: *const T) -> T {
    // It would be semantically correct to implement this via `copy_nonoverlapping`
    // and `MaybeUninit`, as was done before PR #109035. Calling `assume_init`
    // provides enough information to know that this is a typed operation.

    // However, as of March 2023 the compiler was not capable of taking advantage
    // of that information. Thus, the implementation here switched to an intrinsic,
    // which lowers to `_0 = *src` in MIR, to address a few issues:
    //
    // - Using `MaybeUninit::assume_init` after a `copy_nonoverlapping` was not
    //   turning the untyped copy into a typed load. As such, the generated
    //   `load` in LLVM didn't get various metadata, such as `!range` (#73258),
    //   `!nonnull`, and `!noundef`, resulting in poorer optimization.
    // - Going through the extra local resulted in multiple extra copies, even
    //   in optimized MIR.  (Ignoring StorageLive/Dead, the intrinsic is one
    //   MIR statement, while the previous implementation was eight.)  LLVM
    //   could sometimes optimize them away, but because `read` is at the core
    //   of so many things, not having them in the first place improves what we
    //   hand off to the backend.  For example, `mem::replace::<Big>` previously
    //   emitted 4 `alloca` and 6 `memcpy`s, but is now 1 `alloc` and 3 `memcpy`s.
    // - In general, this approach keeps us from getting any more bugs (like
    //   #106369) that boil down to "`read(p)` is worse than `*p`", as this
    //   makes them look identical to the backend (or other MIR consumers).
    //
    // Future enhancements to MIR optimizations might well allow this to return
    // to the previous implementation, rather than using an intrinsic.

    // SAFETY: the caller must guarantee that `src` is valid for reads.
    unsafe {
        #[cfg(debug_assertions)] // Too expensive to always enable (for now?)
        ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "ptr::read requires that the pointer argument is aligned and non-null",
            (
                addr: *const () = src as *const (),
                align: usize = align_of::<T>(),
                is_zst: bool = T::IS_ZST,
            ) => ub_checks::maybe_is_aligned_and_not_null(addr, align, is_zst)
        );
        crate::intrinsics::read_via_copy(src)
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
/// Instead you must use the `&raw const` syntax to create the pointer.
/// You may use that constructed pointer together with this function.
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
/// let unaligned = &raw const packed.unaligned;
///
/// let v = unsafe { std::ptr::read_unaligned(unaligned) };
/// assert_eq!(v, 0x01020304);
/// ```
///
/// Accessing unaligned fields directly with e.g. `packed.unaligned` is safe however.
///
/// # Examples
///
/// Read a `usize` value from a byte buffer:
///
/// ```
/// fn read_usize(x: &[u8]) -> usize {
///     assert!(x.len() >= size_of::<usize>());
///
///     let ptr = x.as_ptr() as *const usize;
///
///     unsafe { ptr.read_unaligned() }
/// }
/// ```
#[inline]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
#[rustc_const_stable(feature = "const_ptr_read", since = "1.71.0")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
#[rustc_diagnostic_item = "ptr_read_unaligned"]
pub const unsafe fn read_unaligned<T>(src: *const T) -> T {
    let mut tmp = MaybeUninit::<T>::uninit();
    // SAFETY: the caller must guarantee that `src` is valid for reads.
    // `src` cannot overlap `tmp` because `tmp` was just allocated on
    // the stack as a separate allocated object.
    //
    // Also, since we just wrote a valid value into `tmp`, it is guaranteed
    // to be properly initialized.
    unsafe {
        copy_nonoverlapping(src as *const u8, tmp.as_mut_ptr() as *mut u8, size_of::<T>());
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
/// Note that even if `T` has size `0`, the pointer must be properly aligned.
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
#[rustc_const_stable(feature = "const_ptr_write", since = "1.83.0")]
#[rustc_diagnostic_item = "ptr_write"]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub const unsafe fn write<T>(dst: *mut T, src: T) {
    // Semantically, it would be fine for this to be implemented as a
    // `copy_nonoverlapping` and appropriate drop suppression of `src`.

    // However, implementing via that currently produces more MIR than is ideal.
    // Using an intrinsic keeps it down to just the simple `*dst = move src` in
    // MIR (11 statements shorter, at the time of writing), and also allows
    // `src` to stay an SSA value in codegen_ssa, rather than a memory one.

    // SAFETY: the caller must guarantee that `dst` is valid for writes.
    // `dst` cannot overlap `src` because the caller has mutable access
    // to `dst` while `src` is owned by this function.
    unsafe {
        #[cfg(debug_assertions)] // Too expensive to always enable (for now?)
        ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "ptr::write requires that the pointer argument is aligned and non-null",
            (
                addr: *mut () = dst as *mut (),
                align: usize = align_of::<T>(),
                is_zst: bool = T::IS_ZST,
            ) => ub_checks::maybe_is_aligned_and_not_null(addr, align, is_zst)
        );
        intrinsics::write_via_move(dst, src)
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
/// Instead, you must use the `&raw mut` syntax to create the pointer.
/// You may use that constructed pointer together with this function.
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
/// let unaligned = &raw mut packed.unaligned;
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
/// Write a `usize` value to a byte buffer:
///
/// ```
/// fn write_usize(x: &mut [u8], val: usize) {
///     assert!(x.len() >= size_of::<usize>());
///
///     let ptr = x.as_mut_ptr() as *mut usize;
///
///     unsafe { ptr.write_unaligned(val) }
/// }
/// ```
#[inline]
#[stable(feature = "ptr_unaligned", since = "1.17.0")]
#[rustc_const_stable(feature = "const_ptr_write", since = "1.83.0")]
#[rustc_diagnostic_item = "ptr_write_unaligned"]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub const unsafe fn write_unaligned<T>(dst: *mut T, src: T) {
    // SAFETY: the caller must guarantee that `dst` is valid for writes.
    // `dst` cannot overlap `src` because the caller has mutable access
    // to `dst` while `src` is owned by this function.
    unsafe {
        copy_nonoverlapping((&raw const src) as *const u8, dst as *mut u8, size_of::<T>());
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
/// Note that even if `T` has size `0`, the pointer must be properly aligned.
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
#[rustc_diagnostic_item = "ptr_read_volatile"]
pub unsafe fn read_volatile<T>(src: *const T) -> T {
    // SAFETY: the caller must uphold the safety contract for `volatile_load`.
    unsafe {
        ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "ptr::read_volatile requires that the pointer argument is aligned and non-null",
            (
                addr: *const () = src as *const (),
                align: usize = align_of::<T>(),
                is_zst: bool = T::IS_ZST,
            ) => ub_checks::maybe_is_aligned_and_not_null(addr, align, is_zst)
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
/// Note that even if `T` has size `0`, the pointer must be properly aligned.
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
#[rustc_diagnostic_item = "ptr_write_volatile"]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn write_volatile<T>(dst: *mut T, src: T) {
    // SAFETY: the caller must uphold the safety contract for `volatile_store`.
    unsafe {
        ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "ptr::write_volatile requires that the pointer argument is aligned and non-null",
            (
                addr: *mut () = dst as *mut (),
                align: usize = align_of::<T>(),
                is_zst: bool = T::IS_ZST,
            ) => ub_checks::maybe_is_aligned_and_not_null(addr, align, is_zst)
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
#[allow(ptr_to_integer_transmute_in_consts)]
pub(crate) unsafe fn align_offset<T: Sized>(p: *const T, a: usize) -> usize {
    // FIXME(#75598): Direct use of these intrinsics improves codegen significantly at opt-level <=
    // 1, where the method versions of these operations are not inlined.
    use intrinsics::{
        assume, cttz_nonzero, exact_div, mul_with_overflow, unchecked_rem, unchecked_shl,
        unchecked_shr, unchecked_sub, wrapping_add, wrapping_mul, wrapping_sub,
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

    let stride = size_of::<T>();

    let addr: usize = p.addr();

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

        let aligned_address = wrapping_add(addr, a_minus_one) & wrapping_sub(0, a);
        let byte_offset = wrapping_sub(aligned_address, addr);
        // FIXME: Remove the assume after <https://github.com/llvm/llvm-project/issues/62502>
        // SAFETY: Masking by `-a` can only affect the low bits, and thus cannot have reduced
        // the value by more than `a-1`, so even though the intermediate values might have
        // wrapped, the byte_offset is always in `[0, a)`.
        unsafe { assume(byte_offset < a) };

        // SAFETY: `stride == 0` case has been handled by the special case above.
        let addr_mod_stride = unsafe { unchecked_rem(addr, stride) };

        return if addr_mod_stride == 0 {
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
    // FIXME(const-hack) replace with min
    let gcdpow = unsafe {
        let x = cttz_nonzero(stride);
        let y = cttz_nonzero(a);
        if x < y { x } else { y }
    };
    // SAFETY: gcdpow has an upper-bound that’s at most the number of bits in a `usize`.
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
#[must_use = "pointer comparison produces a value"]
#[rustc_diagnostic_item = "ptr_eq"]
#[allow(ambiguous_wide_pointer_comparisons)] // it's actually clear here
pub fn eq<T: ?Sized>(a: *const T, b: *const T) -> bool {
    a == b
}

/// Compares the *addresses* of the two pointers for equality,
/// ignoring any metadata in fat pointers.
///
/// If the arguments are thin pointers of the same type,
/// then this is the same as [`eq`].
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// let whole: &[i32; 3] = &[1, 2, 3];
/// let first: &i32 = &whole[0];
///
/// assert!(ptr::addr_eq(whole, first));
/// assert!(!ptr::eq::<dyn std::fmt::Debug>(whole, first));
/// ```
#[stable(feature = "ptr_addr_eq", since = "1.76.0")]
#[inline(always)]
#[must_use = "pointer comparison produces a value"]
pub fn addr_eq<T: ?Sized, U: ?Sized>(p: *const T, q: *const U) -> bool {
    (p as *const ()) == (q as *const ())
}

/// Compares the *addresses* of the two function pointers for equality.
///
/// This is the same as `f == g`, but using this function makes clear that the potentially
/// surprising semantics of function pointer comparison are involved.
///
/// There are **very few guarantees** about how functions are compiled and they have no intrinsic
/// “identity”; in particular, this comparison:
///
/// * May return `true` unexpectedly, in cases where functions are equivalent.
///
///   For example, the following program is likely (but not guaranteed) to print `(true, true)`
///   when compiled with optimization:
///
///   ```
///   let f: fn(i32) -> i32 = |x| x;
///   let g: fn(i32) -> i32 = |x| x + 0;  // different closure, different body
///   let h: fn(u32) -> u32 = |x| x + 0;  // different signature too
///   dbg!(std::ptr::fn_addr_eq(f, g), std::ptr::fn_addr_eq(f, h)); // not guaranteed to be equal
///   ```
///
/// * May return `false` in any case.
///
///   This is particularly likely with generic functions but may happen with any function.
///   (From an implementation perspective, this is possible because functions may sometimes be
///   processed more than once by the compiler, resulting in duplicate machine code.)
///
/// Despite these false positives and false negatives, this comparison can still be useful.
/// Specifically, if
///
/// * `T` is the same type as `U`, `T` is a [subtype] of `U`, or `U` is a [subtype] of `T`, and
/// * `ptr::fn_addr_eq(f, g)` returns true,
///
/// then calling `f` and calling `g` will be equivalent.
///
///
/// # Examples
///
/// ```
/// use std::ptr;
///
/// fn a() { println!("a"); }
/// fn b() { println!("b"); }
/// assert!(!ptr::fn_addr_eq(a as fn(), b as fn()));
/// ```
///
/// [subtype]: https://doc.rust-lang.org/reference/subtyping.html
#[stable(feature = "ptr_fn_addr_eq", since = "1.85.0")]
#[inline(always)]
#[must_use = "function pointer comparison produces a value"]
pub fn fn_addr_eq<T: FnPtr, U: FnPtr>(f: T, g: U) -> bool {
    f.addr() == g.addr()
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
/// use std::hash::{DefaultHasher, Hash, Hasher};
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

#[stable(feature = "fnptr_impls", since = "1.4.0")]
impl<F: FnPtr> PartialEq for F {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.addr() == other.addr()
    }
}
#[stable(feature = "fnptr_impls", since = "1.4.0")]
impl<F: FnPtr> Eq for F {}

#[stable(feature = "fnptr_impls", since = "1.4.0")]
impl<F: FnPtr> PartialOrd for F {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.addr().partial_cmp(&other.addr())
    }
}
#[stable(feature = "fnptr_impls", since = "1.4.0")]
impl<F: FnPtr> Ord for F {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.addr().cmp(&other.addr())
    }
}

#[stable(feature = "fnptr_impls", since = "1.4.0")]
impl<F: FnPtr> hash::Hash for F {
    fn hash<HH: hash::Hasher>(&self, state: &mut HH) {
        state.write_usize(self.addr() as _)
    }
}

#[stable(feature = "fnptr_impls", since = "1.4.0")]
impl<F: FnPtr> fmt::Pointer for F {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::pointer_fmt_inner(self.addr() as _, f)
    }
}

#[stable(feature = "fnptr_impls", since = "1.4.0")]
impl<F: FnPtr> fmt::Debug for F {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::pointer_fmt_inner(self.addr() as _, f)
    }
}

/// Creates a `const` raw pointer to a place, without creating an intermediate reference.
///
/// `addr_of!(expr)` is equivalent to `&raw const expr`. The macro is *soft-deprecated*;
/// use `&raw const` instead.
///
/// It is still an open question under which conditions writing through an `addr_of!`-created
/// pointer is permitted. If the place `expr` evaluates to is based on a raw pointer, then the
/// result of `addr_of!` inherits all permissions from that raw pointer. However, if the place is
/// based on a reference, local variable, or `static`, then until all details are decided, the same
/// rules as for shared references apply: it is UB to write through a pointer created with this
/// operation, except for bytes located inside an `UnsafeCell`. Use `&raw mut` (or [`addr_of_mut`])
/// to create a raw pointer that definitely permits mutation.
///
/// Creating a reference with `&`/`&mut` is only allowed if the pointer is properly aligned
/// and points to initialized data. For cases where those requirements do not hold,
/// raw pointers should be used instead. However, `&expr as *const _` creates a reference
/// before casting it to a raw pointer, and that reference is subject to the same rules
/// as all other references. This macro can create a raw pointer *without* creating
/// a reference first.
///
/// See [`addr_of_mut`] for how to create a pointer to uninitialized data.
/// Doing that with `addr_of` would not make much sense since one could only
/// read the data, and that would be Undefined Behavior.
///
/// # Safety
///
/// The `expr` in `addr_of!(expr)` is evaluated as a place expression, but never loads from the
/// place or requires the place to be dereferenceable. This means that `addr_of!((*ptr).field)`
/// still requires the projection to `field` to be in-bounds, using the same rules as [`offset`].
/// However, `addr_of!(*ptr)` is defined behavior even if `ptr` is null, dangling, or misaligned.
///
/// Note that `Deref`/`Index` coercions (and their mutable counterparts) are applied inside
/// `addr_of!` like everywhere else, in which case a reference is created to call `Deref::deref` or
/// `Index::index`, respectively. The statements above only apply when no such coercions are
/// applied.
///
/// [`offset`]: pointer::offset
///
/// # Example
///
/// **Correct usage: Creating a pointer to unaligned data**
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
/// **Incorrect usage: Out-of-bounds fields projection**
///
/// ```rust,no_run
/// use std::ptr;
///
/// #[repr(C)]
/// struct MyStruct {
///     field1: i32,
///     field2: i32,
/// }
///
/// let ptr: *const MyStruct = ptr::null();
/// let fieldptr = unsafe { ptr::addr_of!((*ptr).field2) }; // Undefined Behavior ⚠️
/// ```
///
/// The field projection `.field2` would offset the pointer by 4 bytes,
/// but the pointer is not in-bounds of an allocation for 4 bytes,
/// so this offset is Undefined Behavior.
/// See the [`offset`] docs for a full list of requirements for inbounds pointer arithmetic; the
/// same requirements apply to field projections, even inside `addr_of!`. (In particular, it makes
/// no difference whether the pointer is null or dangling.)
#[stable(feature = "raw_ref_macros", since = "1.51.0")]
#[rustc_macro_transparency = "semitransparent"]
pub macro addr_of($place:expr) {
    &raw const $place
}

/// Creates a `mut` raw pointer to a place, without creating an intermediate reference.
///
/// `addr_of_mut!(expr)` is equivalent to `&raw mut expr`. The macro is *soft-deprecated*;
/// use `&raw mut` instead.
///
/// Creating a reference with `&`/`&mut` is only allowed if the pointer is properly aligned
/// and points to initialized data. For cases where those requirements do not hold,
/// raw pointers should be used instead. However, `&mut expr as *mut _` creates a reference
/// before casting it to a raw pointer, and that reference is subject to the same rules
/// as all other references. This macro can create a raw pointer *without* creating
/// a reference first.
///
/// # Safety
///
/// The `expr` in `addr_of_mut!(expr)` is evaluated as a place expression, but never loads from the
/// place or requires the place to be dereferenceable. This means that `addr_of_mut!((*ptr).field)`
/// still requires the projection to `field` to be in-bounds, using the same rules as [`offset`].
/// However, `addr_of_mut!(*ptr)` is defined behavior even if `ptr` is null, dangling, or misaligned.
///
/// Note that `Deref`/`Index` coercions (and their mutable counterparts) are applied inside
/// `addr_of_mut!` like everywhere else, in which case a reference is created to call `Deref::deref`
/// or `Index::index`, respectively. The statements above only apply when no such coercions are
/// applied.
///
/// [`offset`]: pointer::offset
///
/// # Examples
///
/// **Correct usage: Creating a pointer to unaligned data**
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
/// **Correct usage: Creating a pointer to uninitialized data**
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
///
/// **Incorrect usage: Out-of-bounds fields projection**
///
/// ```rust,no_run
/// use std::ptr;
///
/// #[repr(C)]
/// struct MyStruct {
///     field1: i32,
///     field2: i32,
/// }
///
/// let ptr: *mut MyStruct = ptr::null_mut();
/// let fieldptr = unsafe { ptr::addr_of_mut!((*ptr).field2) }; // Undefined Behavior ⚠️
/// ```
///
/// The field projection `.field2` would offset the pointer by 4 bytes,
/// but the pointer is not in-bounds of an allocation for 4 bytes,
/// so this offset is Undefined Behavior.
/// See the [`offset`] docs for a full list of requirements for inbounds pointer arithmetic; the
/// same requirements apply to field projections, even inside `addr_of_mut!`. (In particular, it
/// makes no difference whether the pointer is null or dangling.)
#[stable(feature = "raw_ref_macros", since = "1.51.0")]
#[rustc_macro_transparency = "semitransparent"]
pub macro addr_of_mut($place:expr) {
    &raw mut $place
}
