//! Types that pin data to its location in memory.
//!
//! It is sometimes useful to have objects that are guaranteed not to move,
//! in the sense that their placement in memory does not change, and can thus be relied upon.
//! A prime example of such a scenario would be building self-referential structs,
//! as moving an object with pointers to itself will invalidate them, which could cause undefined
//! behavior.
//!
//! At a high level, a <code>[Pin]\<P></code> ensures that the pointee of any pointer type
//! `P` has a stable location in memory, meaning it cannot be moved elsewhere
//! and its memory cannot be deallocated until it gets dropped. We say that the
//! pointee is "pinned". Things get more subtle when discussing types that
//! combine pinned with non-pinned data; [see below](#projections-and-structural-pinning)
//! for more details.
//!
//! By default, all types in Rust are movable. Rust allows passing all types by-value,
//! and common smart-pointer types such as <code>[Box]\<T></code> and <code>[&mut] T</code> allow
//! replacing and moving the values they contain: you can move out of a <code>[Box]\<T></code>,
//! or you can use [`mem::swap`]. <code>[Pin]\<P></code> wraps a pointer type `P`, so
//! <code>[Pin]<[Box]\<T>></code> functions much like a regular <code>[Box]\<T></code>:
//! when a <code>[Pin]<[Box]\<T>></code> gets dropped, so do its contents, and the memory gets
//! deallocated. Similarly, <code>[Pin]<[&mut] T></code> is a lot like <code>[&mut] T</code>.
//! However, <code>[Pin]\<P></code> does not let clients actually obtain a <code>[Box]\<T></code>
//! or <code>[&mut] T</code> to pinned data, which implies that you cannot use operations such
//! as [`mem::swap`]:
//!
//! ```
//! use std::pin::Pin;
//! fn swap_pins<T>(x: Pin<&mut T>, y: Pin<&mut T>) {
//!     // `mem::swap` needs `&mut T`, but we cannot get it.
//!     // We are stuck, we cannot swap the contents of these references.
//!     // We could use `Pin::get_unchecked_mut`, but that is unsafe for a reason:
//!     // we are not allowed to use it for moving things out of the `Pin`.
//! }
//! ```
//!
//! It is worth reiterating that <code>[Pin]\<P></code> does *not* change the fact that a Rust
//! compiler considers all types movable. [`mem::swap`] remains callable for any `T`. Instead,
//! <code>[Pin]\<P></code> prevents certain *values* (pointed to by pointers wrapped in
//! <code>[Pin]\<P></code>) from being moved by making it impossible to call methods that require
//! <code>[&mut] T</code> on them (like [`mem::swap`]).
//!
//! <code>[Pin]\<P></code> can be used to wrap any pointer type `P`, and as such it interacts with
//! [`Deref`] and [`DerefMut`]. A <code>[Pin]\<P></code> where <code>P: [Deref]</code> should be
//! considered as a "`P`-style pointer" to a pinned <code>P::[Target]</code> – so, a
//! <code>[Pin]<[Box]\<T>></code> is an owned pointer to a pinned `T`, and a
//! <code>[Pin]<[Rc]\<T>></code> is a reference-counted pointer to a pinned `T`.
//! For correctness, <code>[Pin]\<P></code> relies on the implementations of [`Deref`] and
//! [`DerefMut`] not to move out of their `self` parameter, and only ever to
//! return a pointer to pinned data when they are called on a pinned pointer.
//!
//! # `Unpin`
//!
//! Many types are always freely movable, even when pinned, because they do not
//! rely on having a stable address. This includes all the basic types (like
//! [`bool`], [`i32`], and references) as well as types consisting solely of these
//! types. Types that do not care about pinning implement the [`Unpin`]
//! auto-trait, which cancels the effect of <code>[Pin]\<P></code>. For <code>T: [Unpin]</code>,
//! <code>[Pin]<[Box]\<T>></code> and <code>[Box]\<T></code> function identically, as do
//! <code>[Pin]<[&mut] T></code> and <code>[&mut] T</code>.
//!
//! Note that pinning and [`Unpin`] only affect the pointed-to type <code>P::[Target]</code>,
//! not the pointer type `P` itself that got wrapped in <code>[Pin]\<P></code>. For example,
//! whether or not <code>[Box]\<T></code> is [`Unpin`] has no effect on the behavior of
//! <code>[Pin]<[Box]\<T>></code> (here, `T` is the pointed-to type).
//!
//! # Example: self-referential struct
//!
//! Before we go into more details to explain the guarantees and choices
//! associated with <code>[Pin]\<P></code>, we discuss some examples for how it might be used.
//! Feel free to [skip to where the theoretical discussion continues](#drop-guarantee).
//!
//! ```rust
//! use std::pin::Pin;
//! use std::marker::PhantomPinned;
//! use std::ptr::NonNull;
//!
//! // This is a self-referential struct because the slice field points to the data field.
//! // We cannot inform the compiler about that with a normal reference,
//! // as this pattern cannot be described with the usual borrowing rules.
//! // Instead we use a raw pointer, though one which is known not to be null,
//! // as we know it's pointing at the string.
//! struct Unmovable {
//!     data: String,
//!     slice: NonNull<String>,
//!     _pin: PhantomPinned,
//! }
//!
//! impl Unmovable {
//!     // To ensure the data doesn't move when the function returns,
//!     // we place it in the heap where it will stay for the lifetime of the object,
//!     // and the only way to access it would be through a pointer to it.
//!     fn new(data: String) -> Pin<Box<Self>> {
//!         let res = Unmovable {
//!             data,
//!             // we only create the pointer once the data is in place
//!             // otherwise it will have already moved before we even started
//!             slice: NonNull::dangling(),
//!             _pin: PhantomPinned,
//!         };
//!         let mut boxed = Box::pin(res);
//!
//!         let slice = NonNull::from(&boxed.data);
//!         // we know this is safe because modifying a field doesn't move the whole struct
//!         unsafe {
//!             let mut_ref: Pin<&mut Self> = Pin::as_mut(&mut boxed);
//!             Pin::get_unchecked_mut(mut_ref).slice = slice;
//!         }
//!         boxed
//!     }
//! }
//!
//! let unmoved = Unmovable::new("hello".to_string());
//! // The pointer should point to the correct location,
//! // so long as the struct hasn't moved.
//! // Meanwhile, we are free to move the pointer around.
//! # #[allow(unused_mut)]
//! let mut still_unmoved = unmoved;
//! assert_eq!(still_unmoved.slice, NonNull::from(&still_unmoved.data));
//!
//! // Since our type doesn't implement Unpin, this will fail to compile:
//! // let mut new_unmoved = Unmovable::new("world".to_string());
//! // std::mem::swap(&mut *still_unmoved, &mut *new_unmoved);
//! ```
//!
//! # Example: intrusive doubly-linked list
//!
//! In an intrusive doubly-linked list, the collection does not actually allocate
//! the memory for the elements itself. Allocation is controlled by the clients,
//! and elements can live on a stack frame that lives shorter than the collection does.
//!
//! To make this work, every element has pointers to its predecessor and successor in
//! the list. Elements can only be added when they are pinned, because moving the elements
//! around would invalidate the pointers. Moreover, the [`Drop`][Drop] implementation of a linked
//! list element will patch the pointers of its predecessor and successor to remove itself
//! from the list.
//!
//! Crucially, we have to be able to rely on [`drop`] being called. If an element
//! could be deallocated or otherwise invalidated without calling [`drop`], the pointers into it
//! from its neighboring elements would become invalid, which would break the data structure.
//!
//! Therefore, pinning also comes with a [`drop`]-related guarantee.
//!
//! # `Drop` guarantee
//!
//! The purpose of pinning is to be able to rely on the placement of some data in memory.
//! To make this work, not just moving the data is restricted; deallocating, repurposing, or
//! otherwise invalidating the memory used to store the data is restricted, too.
//! Concretely, for pinned data you have to maintain the invariant
//! that *its memory will not get invalidated or repurposed from the moment it gets pinned until
//! when [`drop`] is called*.  Only once [`drop`] returns or panics, the memory may be reused.
//!
//! Memory can be "invalidated" by deallocation, but also by
//! replacing a <code>[Some]\(v)</code> by [`None`], or calling [`Vec::set_len`] to "kill" some
//! elements off of a vector. It can be repurposed by using [`ptr::write`] to overwrite it without
//! calling the destructor first. None of this is allowed for pinned data without calling [`drop`].
//!
//! This is exactly the kind of guarantee that the intrusive linked list from the previous
//! section needs to function correctly.
//!
//! Notice that this guarantee does *not* mean that memory does not leak! It is still
//! completely okay to not ever call [`drop`] on a pinned element (e.g., you can still
//! call [`mem::forget`] on a <code>[Pin]<[Box]\<T>></code>). In the example of the doubly-linked
//! list, that element would just stay in the list. However you must not free or reuse the storage
//! *without calling [`drop`]*.
//!
//! # `Drop` implementation
//!
//! If your type uses pinning (such as the two examples above), you have to be careful
//! when implementing [`Drop`][Drop]. The [`drop`] function takes <code>[&mut] self</code>, but this
//! is called *even if your type was previously pinned*! It is as if the
//! compiler automatically called [`Pin::get_unchecked_mut`].
//!
//! This can never cause a problem in safe code because implementing a type that
//! relies on pinning requires unsafe code, but be aware that deciding to make
//! use of pinning in your type (for example by implementing some operation on
//! <code>[Pin]<[&]Self></code> or <code>[Pin]<[&mut] Self></code>) has consequences for your
//! [`Drop`][Drop]implementation as well: if an element of your type could have been pinned,
//! you must treat [`Drop`][Drop] as implicitly taking <code>[Pin]<[&mut] Self></code>.
//!
//! For example, you could implement [`Drop`][Drop] as follows:
//!
//! ```rust,no_run
//! # use std::pin::Pin;
//! # struct Type { }
//! impl Drop for Type {
//!     fn drop(&mut self) {
//!         // `new_unchecked` is okay because we know this value is never used
//!         // again after being dropped.
//!         inner_drop(unsafe { Pin::new_unchecked(self)});
//!         fn inner_drop(this: Pin<&mut Type>) {
//!             // Actual drop code goes here.
//!         }
//!     }
//! }
//! ```
//!
//! The function `inner_drop` has the type that [`drop`] *should* have, so this makes sure that
//! you do not accidentally use `self`/`this` in a way that is in conflict with pinning.
//!
//! Moreover, if your type is `#[repr(packed)]`, the compiler will automatically
//! move fields around to be able to drop them. It might even do
//! that for fields that happen to be sufficiently aligned. As a consequence, you cannot use
//! pinning with a `#[repr(packed)]` type.
//!
//! # Projections and Structural Pinning
//!
//! When working with pinned structs, the question arises how one can access the
//! fields of that struct in a method that takes just <code>[Pin]<[&mut] Struct></code>.
//! The usual approach is to write helper methods (so called *projections*)
//! that turn <code>[Pin]<[&mut] Struct></code> into a reference to the field, but what type should
//! that reference have? Is it <code>[Pin]<[&mut] Field></code> or <code>[&mut] Field</code>?
//! The same question arises with the fields of an `enum`, and also when considering
//! container/wrapper types such as <code>[Vec]\<T></code>, <code>[Box]\<T></code>,
//! or <code>[RefCell]\<T></code>. (This question applies to both mutable and shared references,
//! we just use the more common case of mutable references here for illustration.)
//!
//! It turns out that it is actually up to the author of the data structure to decide whether
//! the pinned projection for a particular field turns <code>[Pin]<[&mut] Struct></code>
//! into <code>[Pin]<[&mut] Field></code> or <code>[&mut] Field</code>. There are some
//! constraints though, and the most important constraint is *consistency*:
//! every field can be *either* projected to a pinned reference, *or* have
//! pinning removed as part of the projection. If both are done for the same field,
//! that will likely be unsound!
//!
//! As the author of a data structure you get to decide for each field whether pinning
//! "propagates" to this field or not. Pinning that propagates is also called "structural",
//! because it follows the structure of the type.
//! In the following subsections, we describe the considerations that have to be made
//! for either choice.
//!
//! ## Pinning *is not* structural for `field`
//!
//! It may seem counter-intuitive that the field of a pinned struct might not be pinned,
//! but that is actually the easiest choice: if a <code>[Pin]<[&mut] Field></code> is never created,
//! nothing can go wrong! So, if you decide that some field does not have structural pinning,
//! all you have to ensure is that you never create a pinned reference to that field.
//!
//! Fields without structural pinning may have a projection method that turns
//! <code>[Pin]<[&mut] Struct></code> into <code>[&mut] Field</code>:
//!
//! ```rust,no_run
//! # use std::pin::Pin;
//! # type Field = i32;
//! # struct Struct { field: Field }
//! impl Struct {
//!     fn pin_get_field(self: Pin<&mut Self>) -> &mut Field {
//!         // This is okay because `field` is never considered pinned.
//!         unsafe { &mut self.get_unchecked_mut().field }
//!     }
//! }
//! ```
//!
//! You may also <code>impl [Unpin] for Struct</code> *even if* the type of `field`
//! is not [`Unpin`]. What that type thinks about pinning is not relevant
//! when no <code>[Pin]<[&mut] Field></code> is ever created.
//!
//! ## Pinning *is* structural for `field`
//!
//! The other option is to decide that pinning is "structural" for `field`,
//! meaning that if the struct is pinned then so is the field.
//!
//! This allows writing a projection that creates a <code>[Pin]<[&mut] Field></code>, thus
//! witnessing that the field is pinned:
//!
//! ```rust,no_run
//! # use std::pin::Pin;
//! # type Field = i32;
//! # struct Struct { field: Field }
//! impl Struct {
//!     fn pin_get_field(self: Pin<&mut Self>) -> Pin<&mut Field> {
//!         // This is okay because `field` is pinned when `self` is.
//!         unsafe { self.map_unchecked_mut(|s| &mut s.field) }
//!     }
//! }
//! ```
//!
//! However, structural pinning comes with a few extra requirements:
//!
//! 1.  The struct must only be [`Unpin`] if all the structural fields are
//!     [`Unpin`]. This is the default, but [`Unpin`] is a safe trait, so as the author of
//!     the struct it is your responsibility *not* to add something like
//!     <code>impl\<T> [Unpin] for Struct\<T></code>. (Notice that adding a projection operation
//!     requires unsafe code, so the fact that [`Unpin`] is a safe trait does not break
//!     the principle that you only have to worry about any of this if you use [`unsafe`].)
//! 2.  The destructor of the struct must not move structural fields out of its argument. This
//!     is the exact point that was raised in the [previous section][drop-impl]: [`drop`] takes
//!     <code>[&mut] self</code>, but the struct (and hence its fields) might have been pinned
//!     before. You have to guarantee that you do not move a field inside your [`Drop`][Drop]
//!     implementation. In particular, as explained previously, this means that your struct
//!     must *not* be `#[repr(packed)]`.
//!     See that section for how to write [`drop`] in a way that the compiler can help you
//!     not accidentally break pinning.
//! 3.  You must make sure that you uphold the [`Drop` guarantee][drop-guarantee]:
//!     once your struct is pinned, the memory that contains the
//!     content is not overwritten or deallocated without calling the content's destructors.
//!     This can be tricky, as witnessed by <code>[VecDeque]\<T></code>: the destructor of
//!     <code>[VecDeque]\<T></code> can fail to call [`drop`] on all elements if one of the
//!     destructors panics. This violates the [`Drop`][Drop] guarantee, because it can lead to
//!     elements being deallocated without their destructor being called.
//!     (<code>[VecDeque]\<T></code> has no pinning projections, so this
//!     does not cause unsoundness.)
//! 4.  You must not offer any other operations that could lead to data being moved out of
//!     the structural fields when your type is pinned. For example, if the struct contains an
//!     <code>[Option]\<T></code> and there is a [`take`][Option::take]-like operation with type
//!     <code>fn([Pin]<[&mut] Struct\<T>>) -> [Option]\<T></code>,
//!     that operation can be used to move a `T` out of a pinned `Struct<T>` – which means
//!     pinning cannot be structural for the field holding this data.
//!
//!     For a more complex example of moving data out of a pinned type,
//!     imagine if <code>[RefCell]\<T></code> had a method
//!     <code>fn get_pin_mut(self: [Pin]<[&mut] Self>) -> [Pin]<[&mut] T></code>.
//!     Then we could do the following:
//!     ```compile_fail
//!     fn exploit_ref_cell<T>(rc: Pin<&mut RefCell<T>>) {
//!         { let p = rc.as_mut().get_pin_mut(); } // Here we get pinned access to the `T`.
//!         let rc_shr: &RefCell<T> = rc.into_ref().get_ref();
//!         let b = rc_shr.borrow_mut();
//!         let content = &mut *b; // And here we have `&mut T` to the same data.
//!     }
//!     ```
//!     This is catastrophic, it means we can first pin the content of the
//!     <code>[RefCell]\<T></code> (using <code>[RefCell]::get_pin_mut</code>) and then move that
//!     content using the mutable reference we got later.
//!
//! ## Examples
//!
//! For a type like <code>[Vec]\<T></code>, both possibilities (structural pinning or not) make
//! sense. A <code>[Vec]\<T></code> with structural pinning could have `get_pin`/`get_pin_mut`
//! methods to get pinned references to elements. However, it could *not* allow calling
//! [`pop`][Vec::pop] on a pinned <code>[Vec]\<T></code> because that would move the (structurally
//! pinned) contents! Nor could it allow [`push`][Vec::push], which might reallocate and thus also
//! move the contents.
//!
//! A <code>[Vec]\<T></code> without structural pinning could
//! <code>impl\<T> [Unpin] for [Vec]\<T></code>, because the contents are never pinned
//! and the <code>[Vec]\<T></code> itself is fine with being moved as well.
//! At that point pinning just has no effect on the vector at all.
//!
//! In the standard library, pointer types generally do not have structural pinning,
//! and thus they do not offer pinning projections. This is why <code>[Box]\<T>: [Unpin]</code>
//! holds for all `T`. It makes sense to do this for pointer types, because moving the
//! <code>[Box]\<T></code> does not actually move the `T`: the <code>[Box]\<T></code> can be freely
//! movable (aka [`Unpin`]) even if the `T` is not. In fact, even <code>[Pin]<[Box]\<T>></code> and
//! <code>[Pin]<[&mut] T></code> are always [`Unpin`] themselves, for the same reason:
//! their contents (the `T`) are pinned, but the pointers themselves can be moved without moving
//! the pinned data. For both <code>[Box]\<T></code> and <code>[Pin]<[Box]\<T>></code>,
//! whether the content is pinned is entirely independent of whether the
//! pointer is pinned, meaning pinning is *not* structural.
//!
//! When implementing a [`Future`] combinator, you will usually need structural pinning
//! for the nested futures, as you need to get pinned references to them to call [`poll`].
//! But if your combinator contains any other data that does not need to be pinned,
//! you can make those fields not structural and hence freely access them with a
//! mutable reference even when you just have <code>[Pin]<[&mut] Self></code> (such as in your own
//! [`poll`] implementation).
//!
//! [Deref]: crate::ops::Deref "ops::Deref"
//! [`Deref`]: crate::ops::Deref "ops::Deref"
//! [Target]: crate::ops::Deref::Target "ops::Deref::Target"
//! [`DerefMut`]: crate::ops::DerefMut "ops::DerefMut"
//! [`mem::swap`]: crate::mem::swap "mem::swap"
//! [`mem::forget`]: crate::mem::forget "mem::forget"
//! [Vec]: ../../std/vec/struct.Vec.html "Vec"
//! [`Vec::set_len`]: ../../std/vec/struct.Vec.html#method.set_len "Vec::set_len"
//! [Box]: ../../std/boxed/struct.Box.html "Box"
//! [Vec::pop]: ../../std/vec/struct.Vec.html#method.pop "Vec::pop"
//! [Vec::push]: ../../std/vec/struct.Vec.html#method.push "Vec::push"
//! [Rc]: ../../std/rc/struct.Rc.html "rc::Rc"
//! [RefCell]: crate::cell::RefCell "cell::RefCell"
//! [`drop`]: Drop::drop
//! [VecDeque]: ../../std/collections/struct.VecDeque.html "collections::VecDeque"
//! [`ptr::write`]: crate::ptr::write "ptr::write"
//! [`Future`]: crate::future::Future "future::Future"
//! [drop-impl]: #drop-implementation
//! [drop-guarantee]: #drop-guarantee
//! [`poll`]: crate::future::Future::poll "future::Future::poll"
//! [&]: reference "shared reference"
//! [&mut]: reference "mutable reference"
//! [`unsafe`]: ../../std/keyword.unsafe.html "keyword unsafe"

#![stable(feature = "pin", since = "1.33.0")]

use crate::cmp::{self, PartialEq, PartialOrd};
use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::marker::{Sized, Unpin};
use crate::ops::{CoerceUnsized, Deref, DerefMut, DispatchFromDyn, Receiver};

/// A pinned pointer.
///
/// This is a wrapper around a kind of pointer which makes that pointer "pin" its
/// value in place, preventing the value referenced by that pointer from being moved
/// unless it implements [`Unpin`].
///
/// *See the [`pin` module] documentation for an explanation of pinning.*
///
/// [`pin` module]: self
//
// Note: the `Clone` derive below causes unsoundness as it's possible to implement
// `Clone` for mutable references.
// See <https://internals.rust-lang.org/t/unsoundness-in-pin/11311> for more details.
#[stable(feature = "pin", since = "1.33.0")]
#[lang = "pin"]
#[fundamental]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Pin<P> {
    pointer: P,
}

// The following implementations aren't derived in order to avoid soundness
// issues. `&self.pointer` should not be accessible to untrusted trait
// implementations.
//
// See <https://internals.rust-lang.org/t/unsoundness-in-pin/11311/73> for more details.

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<P: Deref, Q: Deref> PartialEq<Pin<Q>> for Pin<P>
where
    P::Target: PartialEq<Q::Target>,
{
    fn eq(&self, other: &Pin<Q>) -> bool {
        P::Target::eq(self, other)
    }

    fn ne(&self, other: &Pin<Q>) -> bool {
        P::Target::ne(self, other)
    }
}

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<P: Deref<Target: Eq>> Eq for Pin<P> {}

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<P: Deref, Q: Deref> PartialOrd<Pin<Q>> for Pin<P>
where
    P::Target: PartialOrd<Q::Target>,
{
    fn partial_cmp(&self, other: &Pin<Q>) -> Option<cmp::Ordering> {
        P::Target::partial_cmp(self, other)
    }

    fn lt(&self, other: &Pin<Q>) -> bool {
        P::Target::lt(self, other)
    }

    fn le(&self, other: &Pin<Q>) -> bool {
        P::Target::le(self, other)
    }

    fn gt(&self, other: &Pin<Q>) -> bool {
        P::Target::gt(self, other)
    }

    fn ge(&self, other: &Pin<Q>) -> bool {
        P::Target::ge(self, other)
    }
}

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<P: Deref<Target: Ord>> Ord for Pin<P> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        P::Target::cmp(self, other)
    }
}

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<P: Deref<Target: Hash>> Hash for Pin<P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        P::Target::hash(self, state);
    }
}

impl<P: Deref<Target: Unpin>> Pin<P> {
    /// Construct a new `Pin<P>` around a pointer to some data of a type that
    /// implements [`Unpin`].
    ///
    /// Unlike `Pin::new_unchecked`, this method is safe because the pointer
    /// `P` dereferences to an [`Unpin`] type, which cancels the pinning guarantees.
    #[inline(always)]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    #[stable(feature = "pin", since = "1.33.0")]
    pub const fn new(pointer: P) -> Pin<P> {
        // SAFETY: the value pointed to is `Unpin`, and so has no requirements
        // around pinning.
        unsafe { Pin::new_unchecked(pointer) }
    }

    /// Unwraps this `Pin<P>` returning the underlying pointer.
    ///
    /// This requires that the data inside this `Pin` is [`Unpin`] so that we
    /// can ignore the pinning invariants when unwrapping it.
    #[inline(always)]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    #[stable(feature = "pin_into_inner", since = "1.39.0")]
    pub const fn into_inner(pin: Pin<P>) -> P {
        pin.pointer
    }
}

impl<P: Deref> Pin<P> {
    /// Construct a new `Pin<P>` around a reference to some data of a type that
    /// may or may not implement `Unpin`.
    ///
    /// If `pointer` dereferences to an `Unpin` type, `Pin::new` should be used
    /// instead.
    ///
    /// # Safety
    ///
    /// This constructor is unsafe because we cannot guarantee that the data
    /// pointed to by `pointer` is pinned, meaning that the data will not be moved or
    /// its storage invalidated until it gets dropped. If the constructed `Pin<P>` does
    /// not guarantee that the data `P` points to is pinned, that is a violation of
    /// the API contract and may lead to undefined behavior in later (safe) operations.
    ///
    /// By using this method, you are making a promise about the `P::Deref` and
    /// `P::DerefMut` implementations, if they exist. Most importantly, they
    /// must not move out of their `self` arguments: `Pin::as_mut` and `Pin::as_ref`
    /// will call `DerefMut::deref_mut` and `Deref::deref` *on the pinned pointer*
    /// and expect these methods to uphold the pinning invariants.
    /// Moreover, by calling this method you promise that the reference `P`
    /// dereferences to will not be moved out of again; in particular, it
    /// must not be possible to obtain a `&mut P::Target` and then
    /// move out of that reference (using, for example [`mem::swap`]).
    ///
    /// For example, calling `Pin::new_unchecked` on an `&'a mut T` is unsafe because
    /// while you are able to pin it for the given lifetime `'a`, you have no control
    /// over whether it is kept pinned once `'a` ends:
    /// ```
    /// use std::mem;
    /// use std::pin::Pin;
    ///
    /// fn move_pinned_ref<T>(mut a: T, mut b: T) {
    ///     unsafe {
    ///         let p: Pin<&mut T> = Pin::new_unchecked(&mut a);
    ///         // This should mean the pointee `a` can never move again.
    ///     }
    ///     mem::swap(&mut a, &mut b);
    ///     // The address of `a` changed to `b`'s stack slot, so `a` got moved even
    ///     // though we have previously pinned it! We have violated the pinning API contract.
    /// }
    /// ```
    /// A value, once pinned, must remain pinned forever (unless its type implements `Unpin`).
    ///
    /// Similarly, calling `Pin::new_unchecked` on an `Rc<T>` is unsafe because there could be
    /// aliases to the same data that are not subject to the pinning restrictions:
    /// ```
    /// use std::rc::Rc;
    /// use std::pin::Pin;
    ///
    /// fn move_pinned_rc<T>(mut x: Rc<T>) {
    ///     let pinned = unsafe { Pin::new_unchecked(Rc::clone(&x)) };
    ///     {
    ///         let p: Pin<&T> = pinned.as_ref();
    ///         // This should mean the pointee can never move again.
    ///     }
    ///     drop(pinned);
    ///     let content = Rc::get_mut(&mut x).unwrap();
    ///     // Now, if `x` was the only reference, we have a mutable reference to
    ///     // data that we pinned above, which we could use to move it as we have
    ///     // seen in the previous example. We have violated the pinning API contract.
    ///  }
    ///  ```
    ///
    /// [`mem::swap`]: crate::mem::swap
    #[lang = "new_unchecked"]
    #[inline(always)]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    #[stable(feature = "pin", since = "1.33.0")]
    pub const unsafe fn new_unchecked(pointer: P) -> Pin<P> {
        Pin { pointer }
    }

    /// Gets a pinned shared reference from this pinned pointer.
    ///
    /// This is a generic method to go from `&Pin<Pointer<T>>` to `Pin<&T>`.
    /// It is safe because, as part of the contract of `Pin::new_unchecked`,
    /// the pointee cannot move after `Pin<Pointer<T>>` got created.
    /// "Malicious" implementations of `Pointer::Deref` are likewise
    /// ruled out by the contract of `Pin::new_unchecked`.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn as_ref(&self) -> Pin<&P::Target> {
        // SAFETY: see documentation on this function
        unsafe { Pin::new_unchecked(&*self.pointer) }
    }

    /// Unwraps this `Pin<P>` returning the underlying pointer.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that you will continue to
    /// treat the pointer `P` as pinned after you call this function, so that
    /// the invariants on the `Pin` type can be upheld. If the code using the
    /// resulting `P` does not continue to maintain the pinning invariants that
    /// is a violation of the API contract and may lead to undefined behavior in
    /// later (safe) operations.
    ///
    /// If the underlying data is [`Unpin`], [`Pin::into_inner`] should be used
    /// instead.
    #[inline(always)]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    #[stable(feature = "pin_into_inner", since = "1.39.0")]
    pub const unsafe fn into_inner_unchecked(pin: Pin<P>) -> P {
        pin.pointer
    }
}

impl<P: DerefMut> Pin<P> {
    /// Gets a pinned mutable reference from this pinned pointer.
    ///
    /// This is a generic method to go from `&mut Pin<Pointer<T>>` to `Pin<&mut T>`.
    /// It is safe because, as part of the contract of `Pin::new_unchecked`,
    /// the pointee cannot move after `Pin<Pointer<T>>` got created.
    /// "Malicious" implementations of `Pointer::DerefMut` are likewise
    /// ruled out by the contract of `Pin::new_unchecked`.
    ///
    /// This method is useful when doing multiple calls to functions that consume the pinned type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::pin::Pin;
    ///
    /// # struct Type {}
    /// impl Type {
    ///     fn method(self: Pin<&mut Self>) {
    ///         // do something
    ///     }
    ///
    ///     fn call_method_twice(mut self: Pin<&mut Self>) {
    ///         // `method` consumes `self`, so reborrow the `Pin<&mut Self>` via `as_mut`.
    ///         self.as_mut().method();
    ///         self.as_mut().method();
    ///     }
    /// }
    /// ```
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn as_mut(&mut self) -> Pin<&mut P::Target> {
        // SAFETY: see documentation on this function
        unsafe { Pin::new_unchecked(&mut *self.pointer) }
    }

    /// Assigns a new value to the memory behind the pinned reference.
    ///
    /// This overwrites pinned data, but that is okay: its destructor gets
    /// run before being overwritten, so no pinning guarantee is violated.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn set(&mut self, value: P::Target)
    where
        P::Target: Sized,
    {
        *(self.pointer) = value;
    }
}

impl<'a, T: ?Sized> Pin<&'a T> {
    /// Constructs a new pin by mapping the interior value.
    ///
    /// For example, if you  wanted to get a `Pin` of a field of something,
    /// you could use this to get access to that field in one line of code.
    /// However, there are several gotchas with these "pinning projections";
    /// see the [`pin` module] documentation for further details on that topic.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that the data you return
    /// will not move so long as the argument value does not move (for example,
    /// because it is one of the fields of that value), and also that you do
    /// not move out of the argument you receive to the interior function.
    ///
    /// [`pin` module]: self#projections-and-structural-pinning
    #[stable(feature = "pin", since = "1.33.0")]
    pub unsafe fn map_unchecked<U, F>(self, func: F) -> Pin<&'a U>
    where
        U: ?Sized,
        F: FnOnce(&T) -> &U,
    {
        let pointer = &*self.pointer;
        let new_pointer = func(pointer);

        // SAFETY: the safety contract for `new_unchecked` must be
        // upheld by the caller.
        unsafe { Pin::new_unchecked(new_pointer) }
    }

    /// Gets a shared reference out of a pin.
    ///
    /// This is safe because it is not possible to move out of a shared reference.
    /// It may seem like there is an issue here with interior mutability: in fact,
    /// it *is* possible to move a `T` out of a `&RefCell<T>`. However, this is
    /// not a problem as long as there does not also exist a `Pin<&T>` pointing
    /// to the same data, and `RefCell<T>` does not let you create a pinned reference
    /// to its contents. See the discussion on ["pinning projections"] for further
    /// details.
    ///
    /// Note: `Pin` also implements `Deref` to the target, which can be used
    /// to access the inner value. However, `Deref` only provides a reference
    /// that lives for as long as the borrow of the `Pin`, not the lifetime of
    /// the `Pin` itself. This method allows turning the `Pin` into a reference
    /// with the same lifetime as the original `Pin`.
    ///
    /// ["pinning projections"]: self#projections-and-structural-pinning
    #[inline(always)]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    #[stable(feature = "pin", since = "1.33.0")]
    pub const fn get_ref(self) -> &'a T {
        self.pointer
    }
}

impl<'a, T: ?Sized> Pin<&'a mut T> {
    /// Converts this `Pin<&mut T>` into a `Pin<&T>` with the same lifetime.
    #[inline(always)]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    #[stable(feature = "pin", since = "1.33.0")]
    pub const fn into_ref(self) -> Pin<&'a T> {
        Pin { pointer: self.pointer }
    }

    /// Gets a mutable reference to the data inside of this `Pin`.
    ///
    /// This requires that the data inside this `Pin` is `Unpin`.
    ///
    /// Note: `Pin` also implements `DerefMut` to the data, which can be used
    /// to access the inner value. However, `DerefMut` only provides a reference
    /// that lives for as long as the borrow of the `Pin`, not the lifetime of
    /// the `Pin` itself. This method allows turning the `Pin` into a reference
    /// with the same lifetime as the original `Pin`.
    #[inline(always)]
    #[stable(feature = "pin", since = "1.33.0")]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    pub const fn get_mut(self) -> &'a mut T
    where
        T: Unpin,
    {
        self.pointer
    }

    /// Gets a mutable reference to the data inside of this `Pin`.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that you will never move
    /// the data out of the mutable reference you receive when you call this
    /// function, so that the invariants on the `Pin` type can be upheld.
    ///
    /// If the underlying data is `Unpin`, `Pin::get_mut` should be used
    /// instead.
    #[inline(always)]
    #[stable(feature = "pin", since = "1.33.0")]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    pub const unsafe fn get_unchecked_mut(self) -> &'a mut T {
        self.pointer
    }

    /// Construct a new pin by mapping the interior value.
    ///
    /// For example, if you  wanted to get a `Pin` of a field of something,
    /// you could use this to get access to that field in one line of code.
    /// However, there are several gotchas with these "pinning projections";
    /// see the [`pin` module] documentation for further details on that topic.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that the data you return
    /// will not move so long as the argument value does not move (for example,
    /// because it is one of the fields of that value), and also that you do
    /// not move out of the argument you receive to the interior function.
    ///
    /// [`pin` module]: self#projections-and-structural-pinning
    #[stable(feature = "pin", since = "1.33.0")]
    pub unsafe fn map_unchecked_mut<U, F>(self, func: F) -> Pin<&'a mut U>
    where
        U: ?Sized,
        F: FnOnce(&mut T) -> &mut U,
    {
        // SAFETY: the caller is responsible for not moving the
        // value out of this reference.
        let pointer = unsafe { Pin::get_unchecked_mut(self) };
        let new_pointer = func(pointer);
        // SAFETY: as the value of `this` is guaranteed to not have
        // been moved out, this call to `new_unchecked` is safe.
        unsafe { Pin::new_unchecked(new_pointer) }
    }
}

impl<T: ?Sized> Pin<&'static T> {
    /// Get a pinned reference from a static reference.
    ///
    /// This is safe, because `T` is borrowed for the `'static` lifetime, which
    /// never ends.
    #[unstable(feature = "pin_static_ref", issue = "78186")]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    pub const fn static_ref(r: &'static T) -> Pin<&'static T> {
        // SAFETY: The 'static borrow guarantees the data will not be
        // moved/invalidated until it gets dropped (which is never).
        unsafe { Pin::new_unchecked(r) }
    }
}

impl<'a, P: DerefMut> Pin<&'a mut Pin<P>> {
    /// Gets a pinned mutable reference from this nested pinned pointer.
    ///
    /// This is a generic method to go from `Pin<&mut Pin<Pointer<T>>>` to `Pin<&mut T>`. It is
    /// safe because the existence of a `Pin<Pointer<T>>` ensures that the pointee, `T`, cannot
    /// move in the future, and this method does not enable the pointee to move. "Malicious"
    /// implementations of `P::DerefMut` are likewise ruled out by the contract of
    /// `Pin::new_unchecked`.
    #[unstable(feature = "pin_deref_mut", issue = "86918")]
    #[inline(always)]
    pub fn as_deref_mut(self) -> Pin<&'a mut P::Target> {
        // SAFETY: What we're asserting here is that going from
        //
        //     Pin<&mut Pin<P>>
        //
        // to
        //
        //     Pin<&mut P::Target>
        //
        // is safe.
        //
        // We need to ensure that two things hold for that to be the case:
        //
        // 1) Once we give out a `Pin<&mut P::Target>`, an `&mut P::Target` will not be given out.
        // 2) By giving out a `Pin<&mut P::Target>`, we do not risk of violating `Pin<&mut Pin<P>>`
        //
        // The existence of `Pin<P>` is sufficient to guarantee #1: since we already have a
        // `Pin<P>`, it must already uphold the pinning guarantees, which must mean that
        // `Pin<&mut P::Target>` does as well, since `Pin::as_mut` is safe. We do not have to rely
        // on the fact that P is _also_ pinned.
        //
        // For #2, we need to ensure that code given a `Pin<&mut P::Target>` cannot cause the
        // `Pin<P>` to move? That is not possible, since `Pin<&mut P::Target>` no longer retains
        // any access to the `P` itself, much less the `Pin<P>`.
        unsafe { self.get_unchecked_mut() }.as_mut()
    }
}

impl<T: ?Sized> Pin<&'static mut T> {
    /// Get a pinned mutable reference from a static mutable reference.
    ///
    /// This is safe, because `T` is borrowed for the `'static` lifetime, which
    /// never ends.
    #[unstable(feature = "pin_static_ref", issue = "78186")]
    #[rustc_const_unstable(feature = "const_pin", issue = "76654")]
    pub const fn static_mut(r: &'static mut T) -> Pin<&'static mut T> {
        // SAFETY: The 'static borrow guarantees the data will not be
        // moved/invalidated until it gets dropped (which is never).
        unsafe { Pin::new_unchecked(r) }
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: Deref> Deref for Pin<P> {
    type Target = P::Target;
    fn deref(&self) -> &P::Target {
        Pin::get_ref(Pin::as_ref(self))
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: DerefMut<Target: Unpin>> DerefMut for Pin<P> {
    fn deref_mut(&mut self) -> &mut P::Target {
        Pin::get_mut(Pin::as_mut(self))
    }
}

#[unstable(feature = "receiver_trait", issue = "none")]
impl<P: Receiver> Receiver for Pin<P> {}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: fmt::Debug> fmt::Debug for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.pointer, f)
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: fmt::Display> fmt::Display for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.pointer, f)
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: fmt::Pointer> fmt::Pointer for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.pointer, f)
    }
}

// Note: this means that any impl of `CoerceUnsized` that allows coercing from
// a type that impls `Deref<Target=impl !Unpin>` to a type that impls
// `Deref<Target=Unpin>` is unsound. Any such impl would probably be unsound
// for other reasons, though, so we just need to take care not to allow such
// impls to land in std.
#[stable(feature = "pin", since = "1.33.0")]
impl<P, U> CoerceUnsized<Pin<U>> for Pin<P> where P: CoerceUnsized<U> {}

#[stable(feature = "pin", since = "1.33.0")]
impl<P, U> DispatchFromDyn<Pin<U>> for Pin<P> where P: DispatchFromDyn<U> {}
