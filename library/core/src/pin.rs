//! Types that pin data to a location in memory.
//!
//! It is sometimes useful to be able to rely upon a certain value not being able to *move*,
//! in the sense that its address in memory cannot change. This is useful especially when there
//! are one or more [*pointers*][pointer] pointing at that value. The ability to rely on this
//! guarantee that the value a [pointer] is pointing at (its **pointee**) will
//!
//! 1. Not be *moved* out of its memory location
//! 2. More generally, remain *valid* at that same memory location
//!
//! is called "pinning." We would say that a value which satisfies these guarantees has been
//! "pinned," in that it has been permanently (until the end of its lifespan) attached to its
//! location in memory, as though pinned to a pinboard. Pinning a value is an incredibly useful
//! building block for [`unsafe`] code to be able to reason about whether a raw pointer to the
//! pinned value is still valid. [As we'll see later][drop-guarantee], this is necessarily from the
//! time the value is first pinned until the end of its lifespan. This concept of "pinning" is
//! necessary to implement safe interfaces on top of things like self-referential types and
//! intrusive data structures which cannot currently be modeled in fully safe Rust using only
//! borrow-checked [references][reference].
//!
//! "Pinning" allows us to put a *value* which exists at some location in memory into a state where
//! safe code cannot *move* that value to a different location in memory or otherwise invalidate it
//! at its current location (unless it implements [`Unpin`], which we will
//! [talk about below][self#unpin]). Anything that wants to interact with the pinned value in a way
//! that has the potential to violate these guarantees must promise that it will not actually
//! violate them, using the [`unsafe`] keyword to mark that such a promise is upheld by the user
//! and not the compiler. In this way, we can allow other [`unsafe`] code to rely on any pointers
//! that point to the pinned value to be valid to dereference while it is pinned.
//!
//! Note that as long as you don't use [`unsafe`], it's impossible to create or misuse a pinned
//! value in a way that is unsound. See the documentation of [`Pin<Ptr>`] for more
//! information on the practicalities of how to pin a value and how to use that pinned value from a
//! user's perspective without using [`unsafe`].
//!
//! The rest of this documentation is intended to be the source of truth for users of [`Pin<Ptr>`]
//! that are implementing the [`unsafe`] pieces of an interface that relies on pinning for validity;
//! users of [`Pin<Ptr>`] in safe code do not need to read it in detail.
//!
//! There are several sections to this documentation:
//!
//! * [What is "*moving*"?][what-is-moving]
//! * [What is "pinning"?][what-is-pinning]
//! * [Address sensitivity, AKA "when do we need pinning?"][address-sensitive-values]
//! * [Examples of types with address-sensitive states][address-sensitive-examples]
//!   * [Self-referential struct][self-ref]
//!   * [Intrusive, doubly-linked list][linked-list]
//! * [Subtle details and the `Drop` guarantee][subtle-details]
//!
//! # What is "*moving*"?
//! [what-is-moving]: self#what-is-moving
//!
//! When we say a value is *moved*, we mean that the compiler copies, byte-for-byte, the
//! value from one location to another. In a purely mechanical sense, this is identical to
//! [`Copy`]ing a value from one place in memory to another. In Rust, "move" carries with it the
//! semantics of ownership transfer from one variable to another, which is the key difference
//! between a [`Copy`] and a move. For the purposes of this module's documentation, however, when
//! we write *move* in italics, we mean *specifically* that the value has *moved* in the mechanical
//! sense of being located at a new place in memory.
//!
//! All values in Rust are trivially *moveable*. This means that the address at which a value is
//! located is not necessarily stable in between borrows. The compiler is allowed to *move* a value
//! to a new address without running any code to notify that value that its address
//! has changed. Although the compiler will not insert memory *moves* where no semantic move has
//! occurred, there are many places where a value *may* be moved. For example, when doing
//! assignment or passing a value into a function.
//!
//! ```
//! #[derive(Default)]
//! struct AddrTracker(Option<usize>);
//!
//! impl AddrTracker {
//!     // If we haven't checked the addr of self yet, store the current
//!     // address. If we have, confirm that the current address is the same
//!     // as it was last time, or else panic.
//!     fn check_for_move(&mut self) {
//!         let current_addr = self as *mut Self as usize;
//!         match self.0 {
//!             None => self.0 = Some(current_addr),
//!             Some(prev_addr) => assert_eq!(prev_addr, current_addr),
//!         }
//!     }
//! }
//!
//! // Create a tracker and store the initial address
//! let mut tracker = AddrTracker::default();
//! tracker.check_for_move();
//!
//! // Here we shadow the variable. This carries a semantic move, and may therefore also
//! // come with a mechanical memory *move*
//! let mut tracker = tracker;
//!
//! // May panic!
//! // tracker.check_for_move();
//! ```
//!
//! In this sense, Rust does not guarantee that `check_for_move()` will never panic, because the
//! compiler is permitted to *move* `tracker` in many situations.
//!
//! Common smart-pointer types such as [`Box<T>`] and [`&mut T`] also allow *moving* the underlying
//! *value* they point at: you can move out of a [`Box<T>`], or you can use [`mem::replace`] to
//! move a `T` out of a [`&mut T`]. Therefore, putting a value (such as `tracker` above) behind a
//! pointer isn't enough on its own to ensure that its address does not change.
//!
//! # What is "pinning"?
//! [what-is-pinning]: self#what-is-pinning
//!
//! We say that a value has been *pinned* when it has been put into a state where it is guaranteed
//! to remain *located at the same place in memory* from the time it is pinned until its
//! [`drop`] is called.
//!
//! ## Address-sensitive values, AKA "when we need pinning"
//! [address-sensitive-values]: self#address-sensitive-values-aka-when-we-need-pinning
//!
//! Most values in Rust are entirely okay with being *moved* around at-will.
//! Types for which it is *always* the case that *any* value of that type can be
//! *moved* at-will should implement [`Unpin`], which we will discuss more [below][self#unpin].
//!
//! [`Pin`] is specifically targeted at allowing the implementation of *safe interfaces* around
//! types which have some state during which they become "address-sensitive." A value in such an
//! "address-sensitive" state is *not* okay with being *moved* around at-will. Such a value must
//! stay *un-moved* and valid during the address-sensitive portion of its lifespan because some
//! interface is relying on those invariants to be true in order for its implementation to be sound.
//!
//! As a motivating example of a type which may become address-sensitive, consider a type which
//! contains a pointer to another piece of its own data, *i.e.* a "self-referential" type. In order
//! for such a type to be implemented soundly, the pointer which points into `self`'s data must be
//! proven valid whenever it is accessed. But if that value is *moved*, the pointer will still
//! point to the old address where the value was located and not into the new location of `self`,
//! thus becoming invalid. A key example of such self-referential types are the state machines
//! generated by the compiler to implement [`Future`] for `async fn`s.
//!
//! Such types that have an *address-sensitive* state usually follow a lifecycle
//! that looks something like so:
//!
//! 1. A value is created which can be freely moved around.
//!     * e.g. calling an async function which returns a state machine implementing [`Future`]
//! 2. An operation causes the value to depend on its own address not changing
//!     * e.g. calling [`poll`] for the first time on the produced [`Future`]
//! 3. Further pieces of the safe interface of the type use internal [`unsafe`] operations which
//! assume that the address of the value is stable
//!     * e.g. subsequent calls to [`poll`]
//! 4. Before the value is invalidated (e.g. deallocated), it is *dropped*, giving it a chance to
//! notify anything with pointers to itself that those pointers will be invalidated
//!     * e.g. [`drop`]ping the [`Future`] [^pin-drop-future]
//!
//! There are two possible ways to ensure the invariants required for 2. and 3. above (which
//! apply to any address-sensitive type, not just self-referential types) do not get broken.
//!
//! 1. Have the value detect when it is moved and update all the pointers that point to itself.
//! 2. Guarantee that the address of the value does not change (and that memory is not re-used
//! for anything else) during the time that the pointers to it are expected to be valid to
//! dereference.
//!
//! Since, as we discussed, Rust can move values without notifying them that they have moved, the
//! first option is ruled out.
//!
//! In order to implement the second option, we must in some way enforce its key invariant,
//! *i.e.* prevent the value from being *moved* or otherwise invalidated (you may notice this
//! sounds an awful lot like the definition of *pinning* a value). There a few ways one might be
//! able to enforce this invariant in Rust:
//!
//! 1. Offer a wholly `unsafe` API to interact with the object, thus requiring every caller to
//! uphold the invariant themselves
//! 2. Store the value that must not be moved behind a carefully managed pointer internal to
//! the object
//! 3. Leverage the type system to encode and enforce this invariant by presenting a restricted
//! API surface to interact with *any* object that requires these invariants
//!
//! The first option is quite obviously undesirable, as the [`unsafe`]ty of the interface will
//! become viral throughout all code that interacts with the object.
//!
//! The second option is a viable solution to the problem for some use cases, in particular
//! for self-referential types. Under this model, any type that has an address sensitive state
//! would ultimately store its data in something like a [`Box<T>`], carefully manage internal
//! access to that data to ensure no *moves* or other invalidation occurs, and finally
//! provide a safe interface on top.
//!
//! There are a couple of linked disadvantages to using this model. The most significant is that
//! each individual object must assume it is *on its own* to ensure
//! that its data does not become *moved* or otherwise invalidated. Since there is no shared
//! contract between values of different types, an object cannot assume that others interacting
//! with it will properly respect the invariants around interacting with its data and must
//! therefore protect it from everyone. Because of this, *composition* of address-sensitive types
//! requires at least a level of pointer indirection each time a new object is added to the mix
//! (and, practically, a heap allocation).
//!
//! Although there were other reasons as well, this issue of expensive composition is the key thing
//! that drove Rust towards adopting a different model. It is particularly a problem
//! when one considers, for example, the implications of composing together the [`Future`]s which
//! will eventually make up an asynchronous task (including address-sensitive `async fn` state
//! machines). It is plausible that there could be many layers of [`Future`]s composed together,
//! including multiple layers of `async fn`s handling different parts of a task. It was deemed
//! unacceptable to force indirection and allocation for each layer of composition in this case.
//!
//! [`Pin<Ptr>`] is an implementation of the third option. It allows us to solve the issues
//! discussed with the second option by building a *shared contractual language* around the
//! guarantees of "pinning" data.
//!
//! [^pin-drop-future]: Futures themselves do not ever need to notify other bits of code that
//! they are being dropped, however data structures like stack-based intrusive linked lists do.
//!
//! ## Using [`Pin<Ptr>`] to pin values
//!
//! In order to pin a value, we wrap a *pointer to that value* (of some type `Ptr`) in a
//! [`Pin<Ptr>`]. [`Pin<Ptr>`] can wrap any pointer type, forming a promise that the **pointee**
//! will not be *moved* or [otherwise invalidated][subtle-details].
//!
//! We call such a [`Pin`]-wrapped pointer a **pinning pointer,** (or pinning reference, or pinning
//! `Box`, etc.) because its existence is the thing that is conceptually pinning the underlying
//! pointee in place: it is the metaphorical "pin" securing the data in place on the pinboard
//! (in memory).
//!
//! Notice that the thing wrapped by [`Pin`] is not the value which we want to pin itself, but
//! rather a pointer to that value! A [`Pin<Ptr>`] does not pin the `Ptr`; instead, it pins the
//! pointer's ***pointee** value*.
//!
//! ### Pinning as a library contract
//!
//! Pinning does not require nor make use of any compiler "magic"[^noalias], only a specific
//! contract between the [`unsafe`] parts of a library API and its users.
//!
//! It is important to stress this point as a user of the [`unsafe`] parts of the [`Pin`] API.
//! Practically, this means that performing the mechanics of "pinning" a value by creating a
//! [`Pin<Ptr>`] to it *does not* actually change the way the compiler behaves towards the
//! inner value! It is possible to use incorrect [`unsafe`] code to create a [`Pin<Ptr>`] to a
//! value which does not actually satisfy the invariants that a pinned value must satisfy, and in
//! this way lead to undefined behavior even in (from that point) fully safe code. Similarly, using
//! [`unsafe`], one may get access to a bare [`&mut T`] from a [`Pin<Ptr>`] and
//! use that to invalidly *move* the pinned value out. It is the job of the user of the
//! [`unsafe`] parts of the [`Pin`] API to ensure these invariants are not violated.
//!
//! This differs from e.g. [`UnsafeCell`] which changes the semantics of a program's compiled
//! output. A [`Pin<Ptr>`] is a handle to a value which we have promised we will not move out of,
//! but Rust still considers all values themselves to be fundamentally moveable through, *e.g.*
//! assignment or [`mem::replace`].
//!
//! [^noalias]: There is a bit of nuance here that is still being decided about what the aliasing
//! semantics of `Pin<&mut T>` should be, but this is true as of today.
//!
//! ### How [`Pin`] prevents misuse in safe code
//!
//! In order to accomplish the goal of pinning the pointee value, [`Pin<Ptr>`] restricts access to
//! the wrapped `Ptr` type in safe code. Specifically, [`Pin`] disallows the ability to access
//! the wrapped pointer in ways that would allow the user to *move* the underlying pointee value or
//! otherwise re-use that memory for something else without using [`unsafe`]. For example, a
//! [`Pin<&mut T>`] makes it impossible to obtain the wrapped <code>[&mut] T</code> safely because
//! through that <code>[&mut] T</code> it would be possible to *move* the underlying value out of
//! the pointer with [`mem::replace`], etc.
//!
//! As discussed above, this promise must be upheld manually by [`unsafe`] code which interacts
//! with the [`Pin<Ptr>`] so that other [`unsafe`] code can rely on the pointee value being
//! *un-moved* and valid. Interfaces that operate on values which are in an address-sensitive state
//! accept an argument like <code>[Pin]<[&mut] T></code> or <code>[Pin]<[Box]\<T>></code> to
//! indicate this contract to the caller.
//!
//! [As discussed below][drop-guarantee], opting in to using pinning guarantees in the interface
//! of an address-sensitive type has consequences for the implementation of some safe traits on
//! that type as well.
//!
//! ## Interaction between [`Deref`] and [`Pin<Ptr>`]
//!
//! Since [`Pin<Ptr>`] can wrap any pointer type, it uses [`Deref`] and [`DerefMut`] in
//! order to identify the type of the pinned pointee data and provide (restricted) access to it.
//!
//! A [`Pin<Ptr>`] where [`Ptr: Deref`][Deref] is a "`Ptr`-style pinning pointer" to a pinned
//! [`Ptr::Target`][Target] – so, a <code>[Pin]<[Box]\<T>></code> is an owned, pinning pointer to a
//! pinned `T`, and a <code>[Pin]<[Rc]\<T>></code> is a reference-counted, pinning pointer to a
//! pinned `T`.
//!
//! [`Pin<Ptr>`] also uses the [`<Ptr as Deref>::Target`][Target] type information to modify the
//! interface it is allowed to provide for interacting with that data (for example, when a
//! pinning pointer points at pinned data which implements [`Unpin`], as
//! [discussed below][self#unpin]).
//!
//! [`Pin<Ptr>`] requires that implementations of [`Deref`] and [`DerefMut`] on `Ptr` return a
//! pointer to the pinned data directly and do not *move* out of the `self` parameter during their
//! implementation of [`DerefMut::deref_mut`]. It is unsound for [`unsafe`] code to wrap pointer
//! types with such "malicious" implementations of [`Deref`]; see [`Pin<Ptr>::new_unchecked`] for
//! details.
//!
//! ## Fixing `AddrTracker`
//!
//! The guarantee of a stable address is necessary to make our `AddrTracker` example work. When
//! `check_for_move` sees a <code>[Pin]<&mut AddrTracker></code>, it can safely assume that value
//! will exist at that same address until said value goes out of scope, and thus multiple calls
//! to it *cannot* panic.
//!
//! ```
//! use std::marker::PhantomPinned;
//! use std::pin::Pin;
//! use std::pin::pin;
//!
//! #[derive(Default)]
//! struct AddrTracker {
//!     prev_addr: Option<usize>,
//!     // remove auto-implemented `Unpin` bound to mark this type as having some
//!     // address-sensitive state. This is essential for our expected pinning
//!     // guarantees to work, and is discussed more below.
//!     _pin: PhantomPinned,
//! }
//!
//! impl AddrTracker {
//!     fn check_for_move(self: Pin<&mut Self>) {
//!         let current_addr = &*self as *const Self as usize;
//!         match self.prev_addr {
//!             None => {
//!                 // SAFETY: we do not move out of self
//!                 let self_data_mut = unsafe { self.get_unchecked_mut() };
//!                 self_data_mut.prev_addr = Some(current_addr);
//!             },
//!             Some(prev_addr) => assert_eq!(prev_addr, current_addr),
//!         }
//!     }
//! }
//!
//! // 1. Create the value, not yet in an address-sensitive state
//! let tracker = AddrTracker::default();
//!
//! // 2. Pin the value by putting it behind a pinning pointer, thus putting
//! // it into an address-sensitive state
//! let mut ptr_to_pinned_tracker: Pin<&mut AddrTracker> = pin!(tracker);
//! ptr_to_pinned_tracker.as_mut().check_for_move();
//!
//! // Trying to access `tracker` or pass `ptr_to_pinned_tracker` to anything that requires
//! // mutable access to a non-pinned version of it will no longer compile
//!
//! // 3. We can now assume that the tracker value will never be moved, thus
//! // this will never panic!
//! ptr_to_pinned_tracker.as_mut().check_for_move();
//! ```
//!
//! Note that this invariant is enforced by simply making it impossible to call code that would
//! perform a move on the pinned value. This is the case since the only way to access that pinned
//! value is through the pinning <code>[Pin]<[&mut] T>></code>, which in turn restricts our access.
//!
//! ## [`Unpin`]
//!
//! The vast majority of Rust types have no address-sensitive states. These types
//! implement the [`Unpin`] auto-trait, which cancels the restrictive effects of
//! [`Pin`] when the *pointee* type `T` is [`Unpin`]. When [`T: Unpin`][Unpin],
//! <code>[Pin]<[Box]\<T>></code> functions identically to a non-pinning [`Box<T>`]; similarly,
//! <code>[Pin]<[&mut] T></code> would impose no additional restrictions above a regular
//! [`&mut T`].
//!
//! The idea of this trait is to alleviate the reduced ergonomics of APIs that require the use
//! of [`Pin`] for soundness for some types, but which also want to be used by other types that
//! don't care about pinning. The prime example of such an API is [`Future::poll`]. There are many
//! [`Future`] types that don't care about pinning. These futures can implement [`Unpin`] and
//! therefore get around the pinning related restrictions in the API, while still allowing the
//! subset of [`Future`]s which *do* require pinning to be implemented soundly.
//!
//! Note that the interaction between a [`Pin<Ptr>`] and [`Unpin`] is through the type of the
//! **pointee** value, [`<Ptr as Deref>::Target`][Target]. Whether the `Ptr` type itself
//! implements [`Unpin`] does not affect the behavior of a [`Pin<Ptr>`]. For example, whether or not
//! [`Box`] is [`Unpin`] has no effect on the behavior of <code>[Pin]<[Box]\<T>></code>, because
//! `T` is the type of the pointee value, not [`Box`]. So, whether `T` implements [`Unpin`] is
//! the thing that will affect the behavior of the <code>[Pin]<[Box]\<T>></code>.
//!
//! Builtin types that are [`Unpin`] include all of the primitive types, like [`bool`], [`i32`],
//! and [`f32`], references (<code>[&]T</code> and <code>[&mut] T</code>), etc., as well as many
//! core and standard library types like [`Box<T>`], [`String`], and more.
//! These types are marked [`Unpin`] because they do not have an address-sensitive state like the
//! ones we discussed above. If they did have such a state, those parts of their interface would be
//! unsound without being expressed through pinning, and they would then need to not
//! implement [`Unpin`].
//!
//! The compiler is free to take the conservative stance of marking types as [`Unpin`] so long as
//! all of the types that compose its fields are also [`Unpin`]. This is because if a type
//! implements [`Unpin`], then it is unsound for that type's implementation to rely on
//! pinning-related guarantees for soundness, *even* when viewed through a "pinning" pointer! It is
//! the responsibility of the implementor of a type that relies upon pinning for soundness to
//! ensure that type is *not* marked as [`Unpin`] by adding [`PhantomPinned`] field. This is
//! exactly what we did with our `AddrTracker` example above. Without doing this, you *must not*
//! rely on pinning-related guarantees to apply to your type!
//!
//! If need to truly pin a value of a foreign or built-in type that implements [`Unpin`], you'll
//! need to create your own wrapper type around the [`Unpin`] type you want to pin and then
//! opts-out of [`Unpin`] using [`PhantomPinned`].
//!
//! Exposing access to the inner field which you want to remain pinned must then be carefully
//! considered as well! Remember, exposing a method that gives access to a
//! <code>[Pin]<[&mut] InnerT>></code> where <code>InnerT: [Unpin]</code> would allow safe code to
//! trivially move the inner value out of that pinning pointer, which is precisely what you're
//! seeking to prevent! Exposing a field of a pinned value through a pinning pointer is called
//! "projecting" a pin, and the more general case of deciding in which cases a pin should be able
//! to be projected or not is called "structural pinning." We will go into more detail about this
//! [below][structural-pinning].
//!
//! # Examples of address-sensitive types
//! [address-sensitive-examples]: #examples-of-address-sensitive-types
//!
//! ## A self-referential struct
//! [self-ref]: #a-self-referential-struct
//! [`Unmovable`]: #a-self-referential-struct
//!
//! Self-referential structs are the simplest kind of address-sensitive type.
//!
//! It is often useful for a struct to hold a pointer back into itself, which
//! allows the program to efficiently track subsections of the struct.
//! Below, the `slice` field is a pointer into the `data` field, which
//! we could imagine being used to track a sliding window of `data` in parser
//! code.
//!
//! As mentioned before, this pattern is also used extensively by compiler-generated
//! [`Future`]s.
//!
//! ```rust
//! use std::pin::Pin;
//! use std::marker::PhantomPinned;
//! use std::ptr::NonNull;
//!
//! /// This is a self-referential struct because `self.slice` points into `self.data`.
//! struct Unmovable {
//!     /// Backing buffer.
//!     data: [u8; 64],
//!     /// Points at `self.data` which we know is itself non-null. Raw pointer because we can't do
//!     /// this with a normal reference.
//!     slice: NonNull<[u8]>,
//!     /// Suppress `Unpin` so that this cannot be moved out of a `Pin` once constructed.
//!     _pin: PhantomPinned,
//! }
//!
//! impl Unmovable {
//!     /// Creates a new `Unmovable`.
//!     ///
//!     /// To ensure the data doesn't move we place it on the heap behind a pinning Box.
//!     /// Note that the data is pinned, but the `Pin<Box<Self>>` which is pinning it can
//!     /// itself still be moved. This is important because it means we can return the pinning
//!     /// pointer from the function, which is itself a kind of move!
//!     fn new() -> Pin<Box<Self>> {
//!         let res = Unmovable {
//!             data: [0; 64],
//!             // We only create the pointer once the data is in place
//!             // otherwise it will have already moved before we even started.
//!             slice: NonNull::from(&[]),
//!             _pin: PhantomPinned,
//!         };
//!         // First we put the data in a box, which will be its final resting place
//!         let mut boxed = Box::new(res);
//!
//!         // Then we make the slice field point to the proper part of that boxed data.
//!         // From now on we need to make sure we don't move the boxed data.
//!         boxed.slice = NonNull::from(&boxed.data);
//!
//!         // To do that, we pin the data in place by pointing to it with a pinning
//!         // (`Pin`-wrapped) pointer.
//!         //
//!         // `Box::into_pin` makes existing `Box` pin the data in-place without moving it,
//!         // so we can safely do this now *after* inserting the slice pointer above, but we have
//!         // to take care that we haven't performed any other semantic moves of `res` in between.
//!         let pin = Box::into_pin(boxed);
//!
//!         // Now we can return the pinned (through a pinning Box) data
//!         pin
//!     }
//! }
//!
//! let unmovable: Pin<Box<Unmovable>> = Unmovable::new();
//!
//! // The inner pointee `Unmovable` struct will now never be allowed to move.
//! // Meanwhile, we are free to move the pointer around.
//! # #[allow(unused_mut)]
//! let mut still_unmoved = unmovable;
//! assert_eq!(still_unmoved.slice, NonNull::from(&still_unmoved.data));
//!
//! // We cannot mutably dereference a `Pin<Ptr>` unless the pointee is `Unpin` or we use unsafe.
//! // Since our type doesn't implement `Unpin`, this will fail to compile.
//! // let mut new_unmoved = Unmovable::new();
//! // std::mem::swap(&mut *still_unmoved, &mut *new_unmoved);
//! ```
//!
//! ## An intrusive, doubly-linked list
//! [linked-list]: #an-intrusive-doubly-linked-list
//!
//! In an intrusive doubly-linked list, the collection itself does not own the memory in which
//! each of its elements is stored. Instead, each client is free to allocate space for elements it
//! adds to the list in whichever manner it likes, including on the stack! Elements can live on a
//! stack frame that lives shorter than the collection does provided the elements that live in a
//! given stack frame are removed from the list before going out of scope.
//!
//! To make such an intrusive data structure work, every element stores pointers to its predecessor
//! and successor within its own data, rather than having the list structure itself managing those
//! pointers. It is in this sense that the structure is "intrusive": the details of how an
//! element is stored within the larger structure "intrudes" on the implementation of the element
//! type itself!
//!
//! The full implementation details of such a data structure are outside the scope of this
//! documentation, but we will discuss how [`Pin`] can help to do so.
//!
//! Using such an intrusive pattern, elements may only be added when they are pinned. If we think
//! about the consequences of adding non-pinned values to such a list, this becomes clear:
//!
//! *Moving* or otherwise invalidating an element's data would invalidate the pointers back to it
//! which are stored in the elements ahead and behind it. Thus, in order to soundly dereference
//! the pointers stored to the next and previous elements, we must satisfy the guarantee that
//! nothing has invalidated those pointers (which point to data that we do not own).
//!
//! Moreover, the [`Drop`][Drop] implementation of each element must in some way notify its
//! predecessor and successor elements that it should be removed from the list before it is fully
//! destroyed, otherwise the pointers back to it would again become invalidated.
//!
//! Crucially, this means we have to be able to rely on [`drop`] always being called before an
//! element is invalidated. If an element could be deallocated or otherwise invalidated without
//! calling [`drop`], the pointers to it stored in its neighboring elements would
//! become invalid, which would break the data structure.
//!
//! Therefore, pinning data also comes with [the "`Drop` guarantee"][drop-guarantee].
//!
//! # Subtle details and the `Drop` guarantee
//! [subtle-details]: self#subtle-details-and-the-drop-guarantee
//! [drop-guarantee]: self#subtle-details-and-the-drop-guarantee
//!
//! The purpose of pinning is not *just* to prevent a value from being *moved*, but more
//! generally to be able to rely on the pinned value *remaining valid **at a specific place*** in
//! memory.
//!
//! To do so, pinning a value adds an *additional* invariant that must be upheld in order for use
//! of the pinned data to be valid, on top of the ones that must be upheld for a non-pinned value
//! of the same type to be valid:
//!
//! From the moment a value is pinned by constructing a [`Pin`]ning pointer to it, that value
//! must *remain, **valid***, at that same address in memory, *until its [`drop`] handler is
//! called.*
//!
//! There is some subtlety to this which we have not yet talked about in detail. The invariant
//! described above means that, yes,
//!
//! 1. The value must not be moved out of its location in memory
//!
//! but it also implies that,
//!
//! 2. The memory location that stores the value must not get invalidated or otherwise repurposed
//! during the lifespan of the pinned value until its [`drop`] returns or panics
//!
//! This point is subtle but required for intrusive data structures to be implemented soundly.
//!
//! ## `Drop` guarantee
//!
//! There needs to be a way for a pinned value to notify any code that is relying on its pinned
//! status that it is about to be destroyed. In this way, the dependent code can remove the
//! pinned value's address from its data structures or otherwise change its behavior with the
//! knowledge that it can no longer rely on that value existing at the location it was pinned to.
//!
//! Thus, in any situation where we may want to overwrite a pinned value, that value's [`drop`] must
//! be called beforehand (unless the pinned value implements [`Unpin`], in which case we can ignore
//! all of [`Pin`]'s guarantees, as usual).
//!
//! The most common storage-reuse situations occur when a value on the stack is destroyed as part
//! of a function return and when heap storage is freed. In both cases, [`drop`] gets run for us
//! by Rust when using standard safe code. However, for manual heap allocations or otherwise
//! custom-allocated storage, [`unsafe`] code must make sure to call [`ptr::drop_in_place`] before
//! deallocating and re-using said storage.
//!
//! In addition, storage "re-use"/invalidation can happen even if no storage is (de-)allocated.
//! For example, if we had an [`Option`] which contained a `Some(v)` where `v` is pinned, then `v`
//! would be invalidated by setting that option to `None`.
//!
//! Similarly, if a [`Vec`] was used to store pinned values and [`Vec::set_len`] was used to
//! manually "kill" some elements of a vector, all of the items "killed" would become invalidated,
//! which would be *undefined behavior* if those items were pinned.
//!
//! Both of these cases are somewhat contrived, but it is crucial to remember that [`Pin`]ned data
//! *must* be [`drop`]ped before it is invalidated; not just to prevent memory leaks, but as a
//! matter of soundness. As a corollary, the following code can *never* be made safe:
//!
//! ```rust
//! # use std::mem::ManuallyDrop;
//! # use std::pin::Pin;
//! # struct Type;
//! // Pin something inside a `ManuallyDrop`. This is fine on its own.
//! let mut pin: Pin<Box<ManuallyDrop<Type>>> = Box::pin(ManuallyDrop::new(Type));
//!
//! // However, creating a pinning mutable reference to the type *inside*
//! // the `ManuallyDrop` is not!
//! let inner: Pin<&mut Type> = unsafe {
//!     Pin::map_unchecked_mut(pin.as_mut(), |x| &mut **x)
//! };
//! ```
//!
//! Because [`mem::ManuallyDrop`] inhibits the destructor of `Type`, it won't get run when the
//! <code>[Box]<[ManuallyDrop]\<Type>></code> is dropped, thus violating the drop guarantee of the
//! <code>[Pin]<[&mut] Type>></code>.
//!
//! Of course, *leaking* memory in such a way that its underlying storage will never get invalidated
//! or re-used is still fine: [`mem::forget`]ing a [`Box<T>`] prevents its storage from ever getting
//! re-used, so the [`drop`] guarantee is still satisfied.
//!
//! # Implementing an address-sensitive type.
//!
//! This section goes into detail on important considerations for implementing your own
//! address-sensitive types, which are different from merely using [`Pin<Ptr>`] in a generic
//! way.
//!
//! ## Implementing [`Drop`] for types with address-sensitive states
//! [drop-impl]: self#implementing-drop-for-types-with-address-sensitive-states
//!
//! The [`drop`] function takes [`&mut self`], but this is called *even if that `self` has been
//! pinned*! Implementing [`Drop`] for a type with address-sensitive states, because if `self` was
//! indeed in an address-sensitive state before [`drop`] was called, it is as if the compiler
//! automatically called [`Pin::get_unchecked_mut`].
//!
//! This can never cause a problem in purely safe code because creating a pinning pointer to
//! a type which has an address-sensitive (thus does not implement `Unpin`) requires `unsafe`,
//! but it is important to note that choosing to take advantage of pinning-related guarantees
//! to justify validity in the implementation of your type has consequences for that type's
//! [`Drop`][Drop] implementation as well: if an element of your type could have been pinned,
//! you must treat [`Drop`][Drop] as implicitly taking <code>self: [Pin]<[&mut] Self></code>.
//!
//! You should implement [`Drop`] as follows:
//!
//! ```rust,no_run
//! # use std::pin::Pin;
//! # struct Type;
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
//! The function `inner_drop` has the signature that [`drop`] *should* have in this situation.
//! This makes sure that you do not accidentally use `self`/`this` in a way that is in conflict
//! with pinning's invariants.
//!
//! Moreover, if your type is [`#[repr(packed)]`][packed], the compiler will automatically
//! move fields around to be able to drop them. It might even do
//! that for fields that happen to be sufficiently aligned. As a consequence, you cannot use
//! pinning with a [`#[repr(packed)]`][packed] type.
//!
//! ### Implementing [`Drop`] for pointer types which will be used as [`Pin`]ning pointers
//!
//! It should further be noted that creating a pinning pointer of some type `Ptr` *also* carries
//! with it implications on the way that `Ptr` type must implement [`Drop`]
//! (as well as [`Deref`] and [`DerefMut`])! When implementing a pointer type that may be used as
//! a pinning pointer, you must also take the same care described above not to *move* out of or
//! otherwise invalidate the pointee during [`Drop`], [`Deref`], or [`DerefMut`]
//! implementations.
//!
//! ## "Assigning" pinned data
//!
//! Although in general it is not valid to swap data or assign through a [`Pin<Ptr>`] for the same
//! reason that reusing a pinned object's memory is invalid, it is possible to do validly when
//! implemented with special care for the needs of the exact data structure which is being
//! modified. For example, the assigning function must know how to update all uses of the pinned
//! address (and any other invariants necessary to satisfy validity for that type). For
//! [`Unmovable`] (from the example above), we could write an assignment function like so:
//!
//! ```
//! # use std::pin::Pin;
//! # use std::marker::PhantomPinned;
//! # use std::ptr::NonNull;
//! # struct Unmovable {
//! #     data: [u8; 64],
//! #     slice: NonNull<[u8]>,
//! #     _pin: PhantomPinned,
//! # }
//! #
//! impl Unmovable {
//!     // Copies the contents of `src` into `self`, fixing up the self-pointer
//!     // in the process.
//!     fn assign(self: Pin<&mut Self>, src: Pin<&mut Self>) {
//!         unsafe {
//!             let unpinned_self = Pin::into_inner_unchecked(self);
//!             let unpinned_src = Pin::into_inner_unchecked(src);
//!             *unpinned_self = Self {
//!                 data: unpinned_src.data,
//!                 slice: NonNull::from(&mut []),
//!                 _pin: PhantomPinned,
//!             };
//!
//!             let data_ptr = unpinned_src.data.as_ptr() as *const u8;
//!             let slice_ptr = unpinned_src.slice.as_ptr() as *const u8;
//!             let offset = slice_ptr.offset_from(data_ptr) as usize;
//!             let len = (*unpinned_src.slice.as_ptr()).len();
//!
//!             unpinned_self.slice = NonNull::from(&mut unpinned_self.data[offset..offset+len]);
//!         }
//!     }
//! }
//! ```
//!
//! Even though we can't have the compiler do the assignment for us, it's possible to write
//! such specialized functions for types that might need it.
//!
//! Note that it _is_ possible to assign generically through a [`Pin<Ptr>`] by way of [`Pin::set()`].
//! This does not violate any guarantees, since it will run [`drop`] on the pointee value before
//! assigning the new value. Thus, the [`drop`] implementation still has a chance to perform the
//! necessary notifications to dependent values before the memory location of the original pinned
//! value is overwritten.
//!
//! ## Projections and Structural Pinning
//! [structural-pinning]: self#projections-and-structural-pinning
//!
//! With ordinary structs, it is natural that we want to add *projection* methods that allow
//! borrowing one or more of the inner fields of a struct when the caller has access to a
//! borrow of the whole struct:
//!
//! ```
//! # struct Field;
//! struct Struct {
//!     field: Field,
//!     // ...
//! }
//!
//! impl Struct {
//!     fn field(&mut self) -> &mut Field { &mut self.field }
//! }
//! ```
//!
//! When working with address-sensitive types, it's not obvious what the signature of these
//! functions should be. If `field` takes <code>self: [Pin]<[&mut Struct][&mut]></code>, should it
//! return [`&mut Field`] or <code>[Pin]<[`&mut Field`]></code>? This question also arises with
//! `enum`s and wrapper types like [`Vec<T>`], [`Box<T>`], and [`RefCell<T>`]. (This question
//! applies just as well to shared references, but we'll examine the more common case of mutable
//! references for illustration)
//!
//! It turns out that it's up to the author of `Struct` to decide which type the "projection"
//! should produce. The choice must be *consistent* though: if a pin is projected to a field
//! in one place, then it should very likely not be exposed elsewhere without projecting the
//! pin.
//!
//! As the author of a data structure, you get to decide for each field whether pinning
//! "propagates" to this field or not. Pinning that propagates is also called "structural",
//! because it follows the structure of the type.
//!
//! This choice depends on what guarantees you need from the field for your [`unsafe`] code to work.
//! If the field is itself address-sensitive, or participates in the parent struct's address
//! sensitivity, it will need to be structurally pinned.
//!
//! A useful test is if [`unsafe`] code that consumes <code>[Pin]\<[&mut Struct][&mut]></code>
//! also needs to take note of the address of the field itself, it may be evidence that that field
//! is structurally pinned. Unfortunately, there are no hard-and-fast rules.
//!
//! ### Choosing pinning *not to be* structural for `field`...
//!
//! While counter-intuitive, it's often the easier choice: if you do not expose a
//! <code>[Pin]<[&mut] Field></code>, you do not need to be careful about other code
//! moving out of that field, you just have to ensure is that you never create pinning
//! reference to that field. This does of course also mean that if you decide a field does not
//! have structural pinning, you must not write [`unsafe`] code that assumes (invalidly) that the
//! field *is* structurally pinned!
//!
//! Fields without structural pinning may have a projection method that turns
//! <code>[Pin]<[&mut] Struct></code> into [`&mut Field`]:
//!
//! ```rust,no_run
//! # use std::pin::Pin;
//! # type Field = i32;
//! # struct Struct { field: Field }
//! impl Struct {
//!     fn field(self: Pin<&mut Self>) -> &mut Field {
//!         // This is okay because `field` is never considered pinned, therefore we do not
//!         // need to uphold any pinning guarantees for this field in particular. Of course,
//!         // we must not elsewhere assume this field *is* pinned if we choose to expose
//!         // such a method!
//!         unsafe { &mut self.get_unchecked_mut().field }
//!     }
//! }
//! ```
//!
//! You may also in this situation <code>impl [Unpin] for Struct {}</code> *even if* the type of
//! `field` is not [`Unpin`]. Since we have explicitly chosen not to care about pinning guarantees
//! for `field`, the way `field`'s type interacts with pinning is no longer relevant in the
//! context of its use in `Struct`.
//!
//! ### Choosing pinning *to be* structural for `field`...
//!
//! The other option is to decide that pinning is "structural" for `field`,
//! meaning that if the struct is pinned then so is the field.
//!
//! This allows writing a projection that creates a <code>[Pin]<[`&mut Field`]></code>, thus
//! witnessing that the field is pinned:
//!
//! ```rust,no_run
//! # use std::pin::Pin;
//! # type Field = i32;
//! # struct Struct { field: Field }
//! impl Struct {
//!     fn field(self: Pin<&mut Self>) -> Pin<&mut Field> {
//!         // This is okay because `field` is pinned when `self` is.
//!         unsafe { self.map_unchecked_mut(|s| &mut s.field) }
//!     }
//! }
//! ```
//!
//! Structural pinning comes with a few extra requirements:
//!
//! 1.  *Structural [`Unpin`].* A struct can be [`Unpin`] only if all of its
//!     structurally-pinned fields are, too. This is [`Unpin`]'s behavior by default.
//!     However, as a libray author, it is your responsibility not to write something like
//!     <code>impl\<T> [Unpin] for Struct\<T> {}</code> and then offer a method that provides
//!     structural pinning to an inner field of `T`, which may not be [`Unpin`]! (Adding *any*
//!     projection operation requires unsafe code, so the fact that [`Unpin`] is a safe trait does
//!     not break the principle that you only have to worry about any of this if you use
//!     [`unsafe`])
//!
//! 2.  *Pinned Destruction.* As discussed [above][drop-impl], [`drop`] takes
//!     [`&mut self`], but the struct (and hence its fields) might have been pinned
//!     before. The destructor must be written as if its argument was
//!     <code>self: [Pin]\<[`&mut Self`]></code>, instead.
//!
//!     As a consequence, the struct *must not* be [`#[repr(packed)]`][packed].
//!
//! 3.  *Structural Notice of Destruction.* You must uphold the
//!     [`Drop` guarantee][drop-guarantee]: once your struct is pinned, the struct's storage cannot
//!     be re-used without calling the structurally-pinned fields' destructors, as well.
//!
//!     This can be tricky, as witnessed by [`VecDeque<T>`]: the destructor of [`VecDeque<T>`]
//!     can fail to call [`drop`] on all elements if one of the destructors panics. This violates
//!     the [`Drop` guarantee][drop-guarantee], because it can lead to elements being deallocated
//!     without their destructor being called.
//!
//!     [`VecDeque<T>`] has no pinning projections, so its destructor is sound. If it wanted
//!     to provide such structural pinning, its destructor would need to abort the process if any
//!     of the destructors panicked.
//!
//! 4.  You must not offer any other operations that could lead to data being *moved* out of
//!     the structural fields when your type is pinned. For example, if the struct contains an
//!     [`Option<T>`] and there is a [`take`][Option::take]-like operation with type
//!     <code>fn([Pin]<[&mut Struct\<T>][&mut]>) -> [`Option<T>`]</code>,
//!     then that operation can be used to move a `T` out of a pinned `Struct<T>` – which
//!     means pinning cannot be structural for the field holding this data.
//!
//!     For a more complex example of moving data out of a pinned type,
//!     imagine if [`RefCell<T>`] had a method
//!     <code>fn get_pin_mut(self: [Pin]<[`&mut Self`]>) -> [Pin]<[`&mut T`]></code>.
//!     Then we could do the following:
//!     ```compile_fail
//!     # use std::cell::RefCell;
//!     # use std::pin::Pin;
//!     fn exploit_ref_cell<T>(rc: Pin<&mut RefCell<T>>) {
//!         // Here we get pinned access to the `T`.
//!         let _: Pin<&mut T> = rc.as_mut().get_pin_mut();
//!
//!         // And here we have `&mut T` to the same data.
//!         let shared: &RefCell<T> = rc.into_ref().get_ref();
//!         let borrow = shared.borrow_mut();
//!         let content = &mut *borrow;
//!     }
//!     ```
//!     This is catastrophic: it means we can first pin the content of the
//!     [`RefCell<T>`] (using <code>[RefCell]::get_pin_mut</code>) and then move that
//!     content using the mutable reference we got later.
//!
//! ### Structural Pinning examples
//!
//! For a type like [`Vec<T>`], both possibilities (structural pinning or not) make
//! sense. A [`Vec<T>`] with structural pinning could have `get_pin`/`get_pin_mut`
//! methods to get pinning references to elements. However, it could *not* allow calling
//! [`pop`][Vec::pop] on a pinned [`Vec<T>`] because that would move the (structurally
//! pinned) contents! Nor could it allow [`push`][Vec::push], which might reallocate and thus also
//! move the contents.
//!
//! A [`Vec<T>`] without structural pinning could
//! <code>impl\<T> [Unpin] for [`Vec<T>`]</code>, because the contents are never pinned
//! and the [`Vec<T>`] itself is fine with being moved as well.
//! At that point pinning just has no effect on the vector at all.
//!
//! In the standard library, pointer types generally do not have structural pinning,
//! and thus they do not offer pinning projections. This is why <code>[`Box<T>`]: [Unpin]</code>
//! holds for all `T`. It makes sense to do this for pointer types, because moving the
//! [`Box<T>`] does not actually move the `T`: the [`Box<T>`] can be freely
//! movable (aka [`Unpin`]) even if the `T` is not. In fact, even <code>[Pin]<[`Box<T>`]></code> and
//! <code>[Pin]<[`&mut T`]></code> are always [`Unpin`] themselves, for the same reason:
//! their contents (the `T`) are pinned, but the pointers themselves can be moved without moving
//! the pinned data. For both [`Box<T>`] and <code>[Pin]<[`Box<T>`]></code>,
//! whether the content is pinned is entirely independent of whether the
//! pointer is pinned, meaning pinning is *not* structural.
//!
//! When implementing a [`Future`] combinator, you will usually need structural pinning
//! for the nested futures, as you need to get pinning ([`Pin`]-wrapped) references to them to
//! call [`poll`]. But if your combinator contains any other data that does not need to be pinned,
//! you can make those fields not structural and hence freely access them with a
//! mutable reference even when you just have <code>[Pin]<[`&mut Self`]></code>
//! (such as in your own [`poll`] implementation).
//!
//! [`&mut T`]: &mut
//! [`&mut self`]: &mut
//! [`&mut Self`]: &mut
//! [`&mut Field`]: &mut
//! [Deref]: crate::ops::Deref "ops::Deref"
//! [`Deref`]: crate::ops::Deref "ops::Deref"
//! [Target]: crate::ops::Deref::Target "ops::Deref::Target"
//! [`DerefMut`]: crate::ops::DerefMut "ops::DerefMut"
//! [`mem::swap`]: crate::mem::swap "mem::swap"
//! [`mem::forget`]: crate::mem::forget "mem::forget"
//! [ManuallyDrop]: crate::mem::ManuallyDrop "ManuallyDrop"
//! [RefCell]: crate::cell::RefCell "cell::RefCell"
//! [`drop`]: Drop::drop
//! [`ptr::write`]: crate::ptr::write "ptr::write"
//! [`Future`]: crate::future::Future "future::Future"
//! [drop-impl]: #drop-implementation
//! [drop-guarantee]: #drop-guarantee
//! [`poll`]: crate::future::Future::poll "future::Future::poll"
//! [&]: reference "shared reference"
//! [&mut]: reference "mutable reference"
//! [`unsafe`]: ../../std/keyword.unsafe.html "keyword unsafe"
//! [packed]: https://doc.rust-lang.org/nomicon/other-reprs.html#reprpacked
//! [`std::alloc`]: ../../std/alloc/index.html
//! [`Box<T>`]: ../../std/boxed/struct.Box.html
//! [Box]: ../../std/boxed/struct.Box.html "Box"
//! [`Box`]: ../../std/boxed/struct.Box.html "Box"
//! [`Rc<T>`]: ../../std/rc/struct.Rc.html
//! [Rc]: ../../std/rc/struct.Rc.html "rc::Rc"
//! [`Vec<T>`]: ../../std/vec/struct.Vec.html
//! [Vec]: ../../std/vec/struct.Vec.html "Vec"
//! [`Vec`]: ../../std/vec/struct.Vec.html "Vec"
//! [`Vec::set_len`]: ../../std/vec/struct.Vec.html#method.set_len "Vec::set_len"
//! [Vec::pop]: ../../std/vec/struct.Vec.html#method.pop "Vec::pop"
//! [Vec::push]: ../../std/vec/struct.Vec.html#method.push "Vec::push"
//! [`Vec::set_len`]: ../../std/vec/struct.Vec.html#method.set_len
//! [`VecDeque<T>`]: ../../std/collections/struct.VecDeque.html
//! [VecDeque]: ../../std/collections/struct.VecDeque.html "collections::VecDeque"
//! [`String`]: ../../std/string/struct.String.html "String"

#![stable(feature = "pin", since = "1.33.0")]

use crate::hash::{Hash, Hasher};
use crate::ops::{CoerceUnsized, Deref, DerefMut, DerefPure, DispatchFromDyn, LegacyReceiver};
#[allow(unused_imports)]
use crate::{
    cell::{RefCell, UnsafeCell},
    future::Future,
    marker::PhantomPinned,
    mem, ptr,
};
use crate::{cmp, fmt};

/// A pointer which pins its pointee in place.
///
/// [`Pin`] is a wrapper around some kind of pointer `Ptr` which makes that pointer "pin" its
/// pointee value in place, thus preventing the value referenced by that pointer from being moved
/// or otherwise invalidated at that place in memory unless it implements [`Unpin`].
///
/// *See the [`pin` module] documentation for a more thorough exploration of pinning.*
///
/// ## Pinning values with [`Pin<Ptr>`]
///
/// In order to pin a value, we wrap a *pointer to that value* (of some type `Ptr`) in a
/// [`Pin<Ptr>`]. [`Pin<Ptr>`] can wrap any pointer type, forming a promise that the **pointee**
/// will not be *moved* or [otherwise invalidated][subtle-details]. If the pointee value's type
/// implements [`Unpin`], we are free to disregard these requirements entirely and can wrap any
/// pointer to that value in [`Pin`] directly via [`Pin::new`]. If the pointee value's type does
/// not implement [`Unpin`], then Rust will not let us use the [`Pin::new`] function directly and
/// we'll need to construct a [`Pin`]-wrapped pointer in one of the more specialized manners
/// discussed below.
///
/// We call such a [`Pin`]-wrapped pointer a **pinning pointer** (or pinning ref, or pinning
/// [`Box`], etc.) because its existence is the thing that is pinning the underlying pointee in
/// place: it is the metaphorical "pin" securing the data in place on the pinboard (in memory).
///
/// It is important to stress that the thing in the [`Pin`] is not the value which we want to pin
/// itself, but rather a pointer to that value! A [`Pin<Ptr>`] does not pin the `Ptr` but rather
/// the pointer's ***pointee** value*.
///
/// The most common set of types which require pinning related guarantees for soundness are the
/// compiler-generated state machines that implement [`Future`] for the return value of
/// `async fn`s. These compiler-generated [`Future`]s may contain self-referential pointers, one
/// of the most common use cases for [`Pin`]. More details on this point are provided in the
/// [`pin` module] docs, but suffice it to say they require the guarantees provided by pinning to
/// be implemented soundly.
///
/// This requirement for the implementation of `async fn`s means that the [`Future`] trait
/// requires all calls to [`poll`] to use a <code>self: [Pin]\<&mut Self></code> parameter instead
/// of the usual `&mut self`. Therefore, when manually polling a future, you will need to pin it
/// first.
///
/// You may notice that `async fn`-sourced [`Future`]s are only a small percentage of all
/// [`Future`]s that exist, yet we had to modify the signature of [`poll`] for all [`Future`]s
/// to accommodate them. This is unfortunate, but there is a way that the language attempts to
/// alleviate the extra friction that this API choice incurs: the [`Unpin`] trait.
///
/// The vast majority of Rust types have no reason to ever care about being pinned. These
/// types implement the [`Unpin`] trait, which entirely opts all values of that type out of
/// pinning-related guarantees. For values of these types, pinning a value by pointing to it with a
/// [`Pin<Ptr>`] will have no actual effect.
///
/// The reason this distinction exists is exactly to allow APIs like [`Future::poll`] to take a
/// [`Pin<Ptr>`] as an argument for all types while only forcing [`Future`] types that actually
/// care about pinning guarantees pay the ergonomics cost. For the majority of [`Future`] types
/// that don't have a reason to care about being pinned and therefore implement [`Unpin`], the
/// <code>[Pin]\<&mut Self></code> will act exactly like a regular `&mut Self`, allowing direct
/// access to the underlying value. Only types that *don't* implement [`Unpin`] will be restricted.
///
/// ### Pinning a value of a type that implements [`Unpin`]
///
/// If the type of the value you need to "pin" implements [`Unpin`], you can trivially wrap any
/// pointer to that value in a [`Pin`] by calling [`Pin::new`].
///
/// ```
/// use std::pin::Pin;
///
/// // Create a value of a type that implements `Unpin`
/// let mut unpin_future = std::future::ready(5);
///
/// // Pin it by creating a pinning mutable reference to it (ready to be `poll`ed!)
/// let my_pinned_unpin_future: Pin<&mut _> = Pin::new(&mut unpin_future);
/// ```
///
/// ### Pinning a value inside a [`Box`]
///
/// The simplest and most flexible way to pin a value that does not implement [`Unpin`] is to put
/// that value inside a [`Box`] and then turn that [`Box`] into a "pinning [`Box`]" by wrapping it
/// in a [`Pin`]. You can do both of these in a single step using [`Box::pin`]. Let's see an
/// example of using this flow to pin a [`Future`] returned from calling an `async fn`, a common
/// use case as described above.
///
/// ```
/// use std::pin::Pin;
///
/// async fn add_one(x: u32) -> u32 {
///     x + 1
/// }
///
/// // Call the async function to get a future back
/// let fut = add_one(42);
///
/// // Pin the future inside a pinning box
/// let pinned_fut: Pin<Box<_>> = Box::pin(fut);
/// ```
///
/// If you have a value which is already boxed, for example a [`Box<dyn Future>`][Box], you can pin
/// that value in-place at its current memory address using [`Box::into_pin`].
///
/// ```
/// use std::pin::Pin;
/// use std::future::Future;
///
/// async fn add_one(x: u32) -> u32 {
///     x + 1
/// }
///
/// fn boxed_add_one(x: u32) -> Box<dyn Future<Output = u32>> {
///     Box::new(add_one(x))
/// }
///
/// let boxed_fut = boxed_add_one(42);
///
/// // Pin the future inside the existing box
/// let pinned_fut: Pin<Box<_>> = Box::into_pin(boxed_fut);
/// ```
///
/// There are similar pinning methods offered on the other standard library smart pointer types
/// as well, like [`Rc`] and [`Arc`].
///
/// ### Pinning a value on the stack using [`pin!`]
///
/// There are some situations where it is desirable or even required (for example, in a `#[no_std]`
/// context where you don't have access to the standard library or allocation in general) to
/// pin a value which does not implement [`Unpin`] to its location on the stack. Doing so is
/// possible using the [`pin!`] macro. See its documentation for more.
///
/// ## Layout and ABI
///
/// [`Pin<Ptr>`] is guaranteed to have the same memory layout and ABI[^noalias] as `Ptr`.
///
/// [^noalias]: There is a bit of nuance here that is still being decided about whether the
/// aliasing semantics of `Pin<&mut T>` should be different than `&mut T`, but this is true as of
/// today.
///
/// [`pin!`]: crate::pin::pin "pin!"
/// [`Future`]: crate::future::Future "Future"
/// [`poll`]: crate::future::Future::poll "Future::poll"
/// [`Future::poll`]: crate::future::Future::poll "Future::poll"
/// [`pin` module]: self "pin module"
/// [`Rc`]: ../../std/rc/struct.Rc.html "Rc"
/// [`Arc`]: ../../std/sync/struct.Arc.html "Arc"
/// [Box]: ../../std/boxed/struct.Box.html "Box"
/// [`Box`]: ../../std/boxed/struct.Box.html "Box"
/// [`Box::pin`]: ../../std/boxed/struct.Box.html#method.pin "Box::pin"
/// [`Box::into_pin`]: ../../std/boxed/struct.Box.html#method.into_pin "Box::into_pin"
/// [subtle-details]: self#subtle-details-and-the-drop-guarantee "pin subtle details"
/// [`unsafe`]: ../../std/keyword.unsafe.html "keyword unsafe"
//
// Note: the `Clone` derive below causes unsoundness as it's possible to implement
// `Clone` for mutable references.
// See <https://internals.rust-lang.org/t/unsoundness-in-pin/11311> for more details.
#[stable(feature = "pin", since = "1.33.0")]
#[lang = "pin"]
#[fundamental]
#[repr(transparent)]
#[rustc_pub_transparent]
#[derive(Copy, Clone)]
pub struct Pin<Ptr> {
    // FIXME(#93176): this field is made `#[unstable] #[doc(hidden)] pub` to:
    //   - deter downstream users from accessing it (which would be unsound!),
    //   - let the `pin!` macro access it (such a macro requires using struct
    //     literal syntax in order to benefit from lifetime extension).
    //
    // However, if the `Deref` impl exposes a field with the same name as this
    // field, then the two will collide, resulting in a confusing error when the
    // user attempts to access the field through a `Pin<Ptr>`. Therefore, the
    // name `__pointer` is designed to be unlikely to collide with any other
    // field. Long-term, macro hygiene is expected to offer a more robust
    // alternative, alongside `unsafe` fields.
    #[unstable(feature = "unsafe_pin_internals", issue = "none")]
    #[doc(hidden)]
    pub __pointer: Ptr,
}

// The following implementations aren't derived in order to avoid soundness
// issues. `&self.__pointer` should not be accessible to untrusted trait
// implementations.
//
// See <https://internals.rust-lang.org/t/unsoundness-in-pin/11311/73> for more details.

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<Ptr: Deref, Q: Deref> PartialEq<Pin<Q>> for Pin<Ptr>
where
    Ptr::Target: PartialEq<Q::Target>,
{
    fn eq(&self, other: &Pin<Q>) -> bool {
        Ptr::Target::eq(self, other)
    }

    fn ne(&self, other: &Pin<Q>) -> bool {
        Ptr::Target::ne(self, other)
    }
}

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<Ptr: Deref<Target: Eq>> Eq for Pin<Ptr> {}

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<Ptr: Deref, Q: Deref> PartialOrd<Pin<Q>> for Pin<Ptr>
where
    Ptr::Target: PartialOrd<Q::Target>,
{
    fn partial_cmp(&self, other: &Pin<Q>) -> Option<cmp::Ordering> {
        Ptr::Target::partial_cmp(self, other)
    }

    fn lt(&self, other: &Pin<Q>) -> bool {
        Ptr::Target::lt(self, other)
    }

    fn le(&self, other: &Pin<Q>) -> bool {
        Ptr::Target::le(self, other)
    }

    fn gt(&self, other: &Pin<Q>) -> bool {
        Ptr::Target::gt(self, other)
    }

    fn ge(&self, other: &Pin<Q>) -> bool {
        Ptr::Target::ge(self, other)
    }
}

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<Ptr: Deref<Target: Ord>> Ord for Pin<Ptr> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ptr::Target::cmp(self, other)
    }
}

#[stable(feature = "pin_trait_impls", since = "1.41.0")]
impl<Ptr: Deref<Target: Hash>> Hash for Pin<Ptr> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Ptr::Target::hash(self, state);
    }
}

impl<Ptr: Deref<Target: Unpin>> Pin<Ptr> {
    /// Constructs a new `Pin<Ptr>` around a pointer to some data of a type that
    /// implements [`Unpin`].
    ///
    /// Unlike `Pin::new_unchecked`, this method is safe because the pointer
    /// `Ptr` dereferences to an [`Unpin`] type, which cancels the pinning guarantees.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::pin::Pin;
    ///
    /// let mut val: u8 = 5;
    ///
    /// // Since `val` doesn't care about being moved, we can safely create a "facade" `Pin`
    /// // which will allow `val` to participate in `Pin`-bound apis  without checking that
    /// // pinning guarantees are actually upheld.
    /// let mut pinned: Pin<&mut u8> = Pin::new(&mut val);
    /// ```
    #[inline(always)]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    #[stable(feature = "pin", since = "1.33.0")]
    pub const fn new(pointer: Ptr) -> Pin<Ptr> {
        // SAFETY: the value pointed to is `Unpin`, and so has no requirements
        // around pinning.
        unsafe { Pin::new_unchecked(pointer) }
    }

    /// Unwraps this `Pin<Ptr>`, returning the underlying pointer.
    ///
    /// Doing this operation safely requires that the data pointed at by this pinning pointer
    /// implements [`Unpin`] so that we can ignore the pinning invariants when unwrapping it.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::pin::Pin;
    ///
    /// let mut val: u8 = 5;
    /// let pinned: Pin<&mut u8> = Pin::new(&mut val);
    ///
    /// // Unwrap the pin to get the underlying mutable reference to the value. We can do
    /// // this because `val` doesn't care about being moved, so the `Pin` was just
    /// // a "facade" anyway.
    /// let r = Pin::into_inner(pinned);
    /// assert_eq!(*r, 5);
    /// ```
    #[inline(always)]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    #[stable(feature = "pin_into_inner", since = "1.39.0")]
    pub const fn into_inner(pin: Pin<Ptr>) -> Ptr {
        pin.__pointer
    }
}

impl<Ptr: Deref> Pin<Ptr> {
    /// Constructs a new `Pin<Ptr>` around a reference to some data of a type that
    /// may or may not implement [`Unpin`].
    ///
    /// If `pointer` dereferences to an [`Unpin`] type, [`Pin::new`] should be used
    /// instead.
    ///
    /// # Safety
    ///
    /// This constructor is unsafe because we cannot guarantee that the data
    /// pointed to by `pointer` is pinned. At its core, pinning a value means making the
    /// guarantee that the value's data will not be moved nor have its storage invalidated until
    /// it gets dropped. For a more thorough explanation of pinning, see the [`pin` module docs].
    ///
    /// If the caller that is constructing this `Pin<Ptr>` does not ensure that the data `Ptr`
    /// points to is pinned, that is a violation of the API contract and may lead to undefined
    /// behavior in later (even safe) operations.
    ///
    /// By using this method, you are also making a promise about the [`Deref`] and
    /// [`DerefMut`] implementations of `Ptr`, if they exist. Most importantly, they
    /// must not move out of their `self` arguments: `Pin::as_mut` and `Pin::as_ref`
    /// will call `DerefMut::deref_mut` and `Deref::deref` *on the pointer type `Ptr`*
    /// and expect these methods to uphold the pinning invariants.
    /// Moreover, by calling this method you promise that the reference `Ptr`
    /// dereferences to will not be moved out of again; in particular, it
    /// must not be possible to obtain a `&mut Ptr::Target` and then
    /// move out of that reference (using, for example [`mem::swap`]).
    ///
    /// For example, calling `Pin::new_unchecked` on an `&'a mut T` is unsafe because
    /// while you are able to pin it for the given lifetime `'a`, you have no control
    /// over whether it is kept pinned once `'a` ends, and therefore cannot uphold the
    /// guarantee that a value, once pinned, remains pinned until it is dropped:
    ///
    /// ```
    /// use std::mem;
    /// use std::pin::Pin;
    ///
    /// fn move_pinned_ref<T>(mut a: T, mut b: T) {
    ///     unsafe {
    ///         let p: Pin<&mut T> = Pin::new_unchecked(&mut a);
    ///         // This should mean the pointee `a` can never move again.
    ///     }
    ///     mem::swap(&mut a, &mut b); // Potential UB down the road ⚠️
    ///     // The address of `a` changed to `b`'s stack slot, so `a` got moved even
    ///     // though we have previously pinned it! We have violated the pinning API contract.
    /// }
    /// ```
    /// A value, once pinned, must remain pinned until it is dropped (unless its type implements
    /// `Unpin`). Because `Pin<&mut T>` does not own the value, dropping the `Pin` will not drop
    /// the value and will not end the pinning contract. So moving the value after dropping the
    /// `Pin<&mut T>` is still a violation of the API contract.
    ///
    /// Similarly, calling `Pin::new_unchecked` on an `Rc<T>` is unsafe because there could be
    /// aliases to the same data that are not subject to the pinning restrictions:
    /// ```
    /// use std::rc::Rc;
    /// use std::pin::Pin;
    ///
    /// fn move_pinned_rc<T>(mut x: Rc<T>) {
    ///     // This should mean the pointee can never move again.
    ///     let pin = unsafe { Pin::new_unchecked(Rc::clone(&x)) };
    ///     {
    ///         let p: Pin<&T> = pin.as_ref();
    ///         // ...
    ///     }
    ///     drop(pin);
    ///
    ///     let content = Rc::get_mut(&mut x).unwrap(); // Potential UB down the road ⚠️
    ///     // Now, if `x` was the only reference, we have a mutable reference to
    ///     // data that we pinned above, which we could use to move it as we have
    ///     // seen in the previous example. We have violated the pinning API contract.
    /// }
    /// ```
    ///
    /// ## Pinning of closure captures
    ///
    /// Particular care is required when using `Pin::new_unchecked` in a closure:
    /// `Pin::new_unchecked(&mut var)` where `var` is a by-value (moved) closure capture
    /// implicitly makes the promise that the closure itself is pinned, and that *all* uses
    /// of this closure capture respect that pinning.
    /// ```
    /// use std::pin::Pin;
    /// use std::task::Context;
    /// use std::future::Future;
    ///
    /// fn move_pinned_closure(mut x: impl Future, cx: &mut Context<'_>) {
    ///     // Create a closure that moves `x`, and then internally uses it in a pinned way.
    ///     let mut closure = move || unsafe {
    ///         let _ignore = Pin::new_unchecked(&mut x).poll(cx);
    ///     };
    ///     // Call the closure, so the future can assume it has been pinned.
    ///     closure();
    ///     // Move the closure somewhere else. This also moves `x`!
    ///     let mut moved = closure;
    ///     // Calling it again means we polled the future from two different locations,
    ///     // violating the pinning API contract.
    ///     moved(); // Potential UB ⚠️
    /// }
    /// ```
    /// When passing a closure to another API, it might be moving the closure any time, so
    /// `Pin::new_unchecked` on closure captures may only be used if the API explicitly documents
    /// that the closure is pinned.
    ///
    /// The better alternative is to avoid all that trouble and do the pinning in the outer function
    /// instead (here using the [`pin!`][crate::pin::pin] macro):
    /// ```
    /// use std::pin::pin;
    /// use std::task::Context;
    /// use std::future::Future;
    ///
    /// fn move_pinned_closure(mut x: impl Future, cx: &mut Context<'_>) {
    ///     let mut x = pin!(x);
    ///     // Create a closure that captures `x: Pin<&mut _>`, which is safe to move.
    ///     let mut closure = move || {
    ///         let _ignore = x.as_mut().poll(cx);
    ///     };
    ///     // Call the closure, so the future can assume it has been pinned.
    ///     closure();
    ///     // Move the closure somewhere else.
    ///     let mut moved = closure;
    ///     // Calling it again here is fine (except that we might be polling a future that already
    ///     // returned `Poll::Ready`, but that is a separate problem).
    ///     moved();
    /// }
    /// ```
    ///
    /// [`mem::swap`]: crate::mem::swap
    /// [`pin` module docs]: self
    #[lang = "new_unchecked"]
    #[inline(always)]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    #[stable(feature = "pin", since = "1.33.0")]
    pub const unsafe fn new_unchecked(pointer: Ptr) -> Pin<Ptr> {
        Pin { __pointer: pointer }
    }

    /// Gets a shared reference to the pinned value this [`Pin`] points to.
    ///
    /// This is a generic method to go from `&Pin<Pointer<T>>` to `Pin<&T>`.
    /// It is safe because, as part of the contract of `Pin::new_unchecked`,
    /// the pointee cannot move after `Pin<Pointer<T>>` got created.
    /// "Malicious" implementations of `Pointer::Deref` are likewise
    /// ruled out by the contract of `Pin::new_unchecked`.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn as_ref(&self) -> Pin<&Ptr::Target> {
        // SAFETY: see documentation on this function
        unsafe { Pin::new_unchecked(&*self.__pointer) }
    }
}

// These methods being in a `Ptr: DerefMut` impl block concerns semver stability.
// Currently, calling e.g. `.set()` on a `Pin<&T>` sees that `Ptr: DerefMut`
// doesn't hold, and goes to check for a `.set()` method on `T`. But, if the
// `where Ptr: DerefMut` bound is moved to the method, rustc sees the impl block
// as a valid candidate, and doesn't go on to check other candidates when it
// sees that the bound on the method.
impl<Ptr: DerefMut> Pin<Ptr> {
    /// Gets a mutable reference to the pinned value this `Pin<Ptr>` points to.
    ///
    /// This is a generic method to go from `&mut Pin<Pointer<T>>` to `Pin<&mut T>`.
    /// It is safe because, as part of the contract of `Pin::new_unchecked`,
    /// the pointee cannot move after `Pin<Pointer<T>>` got created.
    /// "Malicious" implementations of `Pointer::DerefMut` are likewise
    /// ruled out by the contract of `Pin::new_unchecked`.
    ///
    /// This method is useful when doing multiple calls to functions that consume the
    /// pinning pointer.
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
    pub fn as_mut(&mut self) -> Pin<&mut Ptr::Target> {
        // SAFETY: see documentation on this function
        unsafe { Pin::new_unchecked(&mut *self.__pointer) }
    }

    /// Gets `Pin<&mut T>` to the underlying pinned value from this nested `Pin`-pointer.
    ///
    /// This is a generic method to go from `Pin<&mut Pin<Pointer<T>>>` to `Pin<&mut T>`. It is
    /// safe because the existence of a `Pin<Pointer<T>>` ensures that the pointee, `T`, cannot
    /// move in the future, and this method does not enable the pointee to move. "Malicious"
    /// implementations of `Ptr::DerefMut` are likewise ruled out by the contract of
    /// `Pin::new_unchecked`.
    #[stable(feature = "pin_deref_mut", since = "1.84.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline(always)]
    pub fn as_deref_mut(self: Pin<&mut Pin<Ptr>>) -> Pin<&mut Ptr::Target> {
        // SAFETY: What we're asserting here is that going from
        //
        //     Pin<&mut Pin<Ptr>>
        //
        // to
        //
        //     Pin<&mut Ptr::Target>
        //
        // is safe.
        //
        // We need to ensure that two things hold for that to be the case:
        //
        // 1) Once we give out a `Pin<&mut Ptr::Target>`, a `&mut Ptr::Target` will not be given out.
        // 2) By giving out a `Pin<&mut Ptr::Target>`, we do not risk violating
        // `Pin<&mut Pin<Ptr>>`
        //
        // The existence of `Pin<Ptr>` is sufficient to guarantee #1: since we already have a
        // `Pin<Ptr>`, it must already uphold the pinning guarantees, which must mean that
        // `Pin<&mut Ptr::Target>` does as well, since `Pin::as_mut` is safe. We do not have to rely
        // on the fact that `Ptr` is _also_ pinned.
        //
        // For #2, we need to ensure that code given a `Pin<&mut Ptr::Target>` cannot cause the
        // `Pin<Ptr>` to move? That is not possible, since `Pin<&mut Ptr::Target>` no longer retains
        // any access to the `Ptr` itself, much less the `Pin<Ptr>`.
        unsafe { self.get_unchecked_mut() }.as_mut()
    }

    /// Assigns a new value to the memory location pointed to by the `Pin<Ptr>`.
    ///
    /// This overwrites pinned data, but that is okay: the original pinned value's destructor gets
    /// run before being overwritten and the new value is also a valid value of the same type, so
    /// no pinning invariant is violated. See [the `pin` module documentation][subtle-details]
    /// for more information on how this upholds the pinning invariants.
    ///
    /// # Example
    ///
    /// ```
    /// use std::pin::Pin;
    ///
    /// let mut val: u8 = 5;
    /// let mut pinned: Pin<&mut u8> = Pin::new(&mut val);
    /// println!("{}", pinned); // 5
    /// pinned.set(10);
    /// println!("{}", pinned); // 10
    /// ```
    ///
    /// [subtle-details]: self#subtle-details-and-the-drop-guarantee
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn set(&mut self, value: Ptr::Target)
    where
        Ptr::Target: Sized,
    {
        *(self.__pointer) = value;
    }
}

impl<Ptr: Deref> Pin<Ptr> {
    /// Unwraps this `Pin<Ptr>`, returning the underlying `Ptr`.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that you will continue to
    /// treat the pointer `Ptr` as pinned after you call this function, so that
    /// the invariants on the `Pin` type can be upheld. If the code using the
    /// resulting `Ptr` does not continue to maintain the pinning invariants that
    /// is a violation of the API contract and may lead to undefined behavior in
    /// later (safe) operations.
    ///
    /// Note that you must be able to guarantee that the data pointed to by `Ptr`
    /// will be treated as pinned all the way until its `drop` handler is complete!
    ///
    /// *For more information, see the [`pin` module docs][self]*
    ///
    /// If the underlying data is [`Unpin`], [`Pin::into_inner`] should be used
    /// instead.
    #[inline(always)]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    #[stable(feature = "pin_into_inner", since = "1.39.0")]
    pub const unsafe fn into_inner_unchecked(pin: Pin<Ptr>) -> Ptr {
        pin.__pointer
    }
}

impl<'a, T: ?Sized> Pin<&'a T> {
    /// Constructs a new pin by mapping the interior value.
    ///
    /// For example, if you wanted to get a `Pin` of a field of something,
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
        let pointer = &*self.__pointer;
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
    /// to the inner `T` inside the `RefCell`, and `RefCell<T>` does not let you get a
    /// `Pin<&T>` pointer to its contents. See the discussion on ["pinning projections"]
    /// for further details.
    ///
    /// Note: `Pin` also implements `Deref` to the target, which can be used
    /// to access the inner value. However, `Deref` only provides a reference
    /// that lives for as long as the borrow of the `Pin`, not the lifetime of
    /// the reference contained in the `Pin`. This method allows turning the `Pin` into a reference
    /// with the same lifetime as the reference it wraps.
    ///
    /// ["pinning projections"]: self#projections-and-structural-pinning
    #[inline(always)]
    #[must_use]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    #[stable(feature = "pin", since = "1.33.0")]
    pub const fn get_ref(self) -> &'a T {
        self.__pointer
    }
}

impl<'a, T: ?Sized> Pin<&'a mut T> {
    /// Converts this `Pin<&mut T>` into a `Pin<&T>` with the same lifetime.
    #[inline(always)]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    #[stable(feature = "pin", since = "1.33.0")]
    pub const fn into_ref(self) -> Pin<&'a T> {
        Pin { __pointer: self.__pointer }
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
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "pin", since = "1.33.0")]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    pub const fn get_mut(self) -> &'a mut T
    where
        T: Unpin,
    {
        self.__pointer
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
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "pin", since = "1.33.0")]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    pub const unsafe fn get_unchecked_mut(self) -> &'a mut T {
        self.__pointer
    }

    /// Constructs a new pin by mapping the interior value.
    ///
    /// For example, if you wanted to get a `Pin` of a field of something,
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
    #[must_use = "`self` will be dropped if the result is not used"]
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
    /// Gets a pinning reference from a `&'static` reference.
    ///
    /// This is safe because `T` is borrowed immutably for the `'static` lifetime, which
    /// never ends.
    #[stable(feature = "pin_static_ref", since = "1.61.0")]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    pub const fn static_ref(r: &'static T) -> Pin<&'static T> {
        // SAFETY: The 'static borrow guarantees the data will not be
        // moved/invalidated until it gets dropped (which is never).
        unsafe { Pin::new_unchecked(r) }
    }
}

impl<T: ?Sized> Pin<&'static mut T> {
    /// Gets a pinning mutable reference from a static mutable reference.
    ///
    /// This is safe because `T` is borrowed for the `'static` lifetime, which
    /// never ends.
    #[stable(feature = "pin_static_ref", since = "1.61.0")]
    #[rustc_const_stable(feature = "const_pin", since = "1.84.0")]
    pub const fn static_mut(r: &'static mut T) -> Pin<&'static mut T> {
        // SAFETY: The 'static borrow guarantees the data will not be
        // moved/invalidated until it gets dropped (which is never).
        unsafe { Pin::new_unchecked(r) }
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<Ptr: Deref> Deref for Pin<Ptr> {
    type Target = Ptr::Target;
    fn deref(&self) -> &Ptr::Target {
        Pin::get_ref(Pin::as_ref(self))
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<Ptr: DerefMut<Target: Unpin>> DerefMut for Pin<Ptr> {
    fn deref_mut(&mut self) -> &mut Ptr::Target {
        Pin::get_mut(Pin::as_mut(self))
    }
}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<Ptr: DerefPure> DerefPure for Pin<Ptr> {}

#[unstable(feature = "legacy_receiver_trait", issue = "none")]
impl<Ptr: LegacyReceiver> LegacyReceiver for Pin<Ptr> {}

#[stable(feature = "pin", since = "1.33.0")]
impl<Ptr: fmt::Debug> fmt::Debug for Pin<Ptr> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.__pointer, f)
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<Ptr: fmt::Display> fmt::Display for Pin<Ptr> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.__pointer, f)
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<Ptr: fmt::Pointer> fmt::Pointer for Pin<Ptr> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.__pointer, f)
    }
}

// Note: this means that any impl of `CoerceUnsized` that allows coercing from
// a type that impls `Deref<Target=impl !Unpin>` to a type that impls
// `Deref<Target=Unpin>` is unsound. Any such impl would probably be unsound
// for other reasons, though, so we just need to take care not to allow such
// impls to land in std.
#[stable(feature = "pin", since = "1.33.0")]
impl<Ptr, U> CoerceUnsized<Pin<U>> for Pin<Ptr>
where
    Ptr: CoerceUnsized<U> + PinCoerceUnsized,
    U: PinCoerceUnsized,
{
}

#[stable(feature = "pin", since = "1.33.0")]
impl<Ptr, U> DispatchFromDyn<Pin<U>> for Pin<Ptr>
where
    Ptr: DispatchFromDyn<U> + PinCoerceUnsized,
    U: PinCoerceUnsized,
{
}

#[unstable(feature = "pin_coerce_unsized_trait", issue = "123430")]
/// Trait that indicates that this is a pointer or a wrapper for one, where
/// unsizing can be performed on the pointee when it is pinned.
///
/// # Safety
///
/// If this type implements `Deref`, then the concrete type returned by `deref`
/// and `deref_mut` must not change without a modification. The following
/// operations are not considered modifications:
///
/// * Moving the pointer.
/// * Performing unsizing coercions on the pointer.
/// * Performing dynamic dispatch with the pointer.
/// * Calling `deref` or `deref_mut` on the pointer.
///
/// The concrete type of a trait object is the type that the vtable corresponds
/// to. The concrete type of a slice is an array of the same element type and
/// the length specified in the metadata. The concrete type of a sized type
/// is the type itself.
pub unsafe trait PinCoerceUnsized {}

#[stable(feature = "pin", since = "1.33.0")]
unsafe impl<'a, T: ?Sized> PinCoerceUnsized for &'a T {}

#[stable(feature = "pin", since = "1.33.0")]
unsafe impl<'a, T: ?Sized> PinCoerceUnsized for &'a mut T {}

#[stable(feature = "pin", since = "1.33.0")]
unsafe impl<T: PinCoerceUnsized> PinCoerceUnsized for Pin<T> {}

#[stable(feature = "pin", since = "1.33.0")]
unsafe impl<T: ?Sized> PinCoerceUnsized for *const T {}

#[stable(feature = "pin", since = "1.33.0")]
unsafe impl<T: ?Sized> PinCoerceUnsized for *mut T {}

/// Constructs a <code>[Pin]<[&mut] T></code>, by pinning a `value: T` locally.
///
/// Unlike [`Box::pin`], this does not create a new heap allocation. As explained
/// below, the element might still end up on the heap however.
///
/// The local pinning performed by this macro is usually dubbed "stack"-pinning.
/// Outside of `async` contexts locals do indeed get stored on the stack. In
/// `async` functions or blocks however, any locals crossing an `.await` point
/// are part of the state captured by the `Future`, and will use the storage of
/// those. That storage can either be on the heap or on the stack. Therefore,
/// local pinning is a more accurate term.
///
/// If the type of the given value does not implement [`Unpin`], then this macro
/// pins the value in memory in a way that prevents moves. On the other hand,
/// if the type does implement [`Unpin`], <code>[Pin]<[&mut] T></code> behaves
/// like <code>[&mut] T</code>, and operations such as
/// [`mem::replace()`][crate::mem::replace] or [`mem::take()`](crate::mem::take)
/// will allow moves of the value.
/// See [the `Unpin` section of the `pin` module][self#unpin] for details.
///
/// ## Examples
///
/// ### Basic usage
///
/// ```rust
/// # use core::marker::PhantomPinned as Foo;
/// use core::pin::{pin, Pin};
///
/// fn stuff(foo: Pin<&mut Foo>) {
///     // …
///     # let _ = foo;
/// }
///
/// let pinned_foo = pin!(Foo { /* … */ });
/// stuff(pinned_foo);
/// // or, directly:
/// stuff(pin!(Foo { /* … */ }));
/// ```
///
/// ### Manually polling a `Future` (without `Unpin` bounds)
///
/// ```rust
/// use std::{
///     future::Future,
///     pin::pin,
///     task::{Context, Poll},
///     thread,
/// };
/// # use std::{sync::Arc, task::Wake, thread::Thread};
///
/// # /// A waker that wakes up the current thread when called.
/// # struct ThreadWaker(Thread);
/// #
/// # impl Wake for ThreadWaker {
/// #     fn wake(self: Arc<Self>) {
/// #         self.0.unpark();
/// #     }
/// # }
/// #
/// /// Runs a future to completion.
/// fn block_on<Fut: Future>(fut: Fut) -> Fut::Output {
///     let waker_that_unparks_thread = // …
///         # Arc::new(ThreadWaker(thread::current())).into();
///     let mut cx = Context::from_waker(&waker_that_unparks_thread);
///     // Pin the future so it can be polled.
///     let mut pinned_fut = pin!(fut);
///     loop {
///         match pinned_fut.as_mut().poll(&mut cx) {
///             Poll::Pending => thread::park(),
///             Poll::Ready(res) => return res,
///         }
///     }
/// }
/// #
/// # assert_eq!(42, block_on(async { 42 }));
/// ```
///
/// ### With `Coroutine`s
///
/// ```rust
/// #![feature(coroutines)]
/// #![feature(coroutine_trait)]
/// use core::{
///     ops::{Coroutine, CoroutineState},
///     pin::pin,
/// };
///
/// fn coroutine_fn() -> impl Coroutine<Yield = usize, Return = ()> /* not Unpin */ {
///  // Allow coroutine to be self-referential (not `Unpin`)
///  // vvvvvv        so that locals can cross yield points.
///     #[coroutine] static || {
///         let foo = String::from("foo");
///         let foo_ref = &foo; // ------+
///         yield 0;                  // | <- crosses yield point!
///         println!("{foo_ref}"); // <--+
///         yield foo.len();
///     }
/// }
///
/// fn main() {
///     let mut coroutine = pin!(coroutine_fn());
///     match coroutine.as_mut().resume(()) {
///         CoroutineState::Yielded(0) => {},
///         _ => unreachable!(),
///     }
///     match coroutine.as_mut().resume(()) {
///         CoroutineState::Yielded(3) => {},
///         _ => unreachable!(),
///     }
///     match coroutine.resume(()) {
///         CoroutineState::Yielded(_) => unreachable!(),
///         CoroutineState::Complete(()) => {},
///     }
/// }
/// ```
///
/// ## Remarks
///
/// Precisely because a value is pinned to local storage, the resulting <code>[Pin]<[&mut] T></code>
/// reference ends up borrowing a local tied to that block: it can't escape it.
///
/// The following, for instance, fails to compile:
///
/// ```rust,compile_fail
/// use core::pin::{pin, Pin};
/// # use core::{marker::PhantomPinned as Foo, mem::drop as stuff};
///
/// let x: Pin<&mut Foo> = {
///     let x: Pin<&mut Foo> = pin!(Foo { /* … */ });
///     x
/// }; // <- Foo is dropped
/// stuff(x); // Error: use of dropped value
/// ```
///
/// <details><summary>Error message</summary>
///
/// ```console
/// error[E0716]: temporary value dropped while borrowed
///   --> src/main.rs:9:28
///    |
/// 8  | let x: Pin<&mut Foo> = {
///    |     - borrow later stored here
/// 9  |     let x: Pin<&mut Foo> = pin!(Foo { /* … */ });
///    |                            ^^^^^^^^^^^^^^^^^^^^^ creates a temporary value which is freed while still in use
/// 10 |     x
/// 11 | }; // <- Foo is dropped
///    | - temporary value is freed at the end of this statement
///    |
///    = note: consider using a `let` binding to create a longer lived value
/// ```
///
/// </details>
///
/// This makes [`pin!`] **unsuitable to pin values when intending to _return_ them**. Instead, the
/// value is expected to be passed around _unpinned_ until the point where it is to be consumed,
/// where it is then useful and even sensible to pin the value locally using [`pin!`].
///
/// If you really need to return a pinned value, consider using [`Box::pin`] instead.
///
/// On the other hand, local pinning using [`pin!`] is likely to be cheaper than
/// pinning into a fresh heap allocation using [`Box::pin`]. Moreover, by virtue of not
/// requiring an allocator, [`pin!`] is the main non-`unsafe` `#![no_std]`-compatible [`Pin`]
/// constructor.
///
/// [`Box::pin`]: ../../std/boxed/struct.Box.html#method.pin
#[stable(feature = "pin_macro", since = "1.68.0")]
#[rustc_macro_transparency = "semitransparent"]
#[allow_internal_unstable(unsafe_pin_internals)]
pub macro pin($value:expr $(,)?) {
    // This is `Pin::new_unchecked(&mut { $value })`, so, for starters, let's
    // review such a hypothetical macro (that any user-code could define):
    //
    // ```rust
    // macro_rules! pin {( $value:expr ) => (
    //     match &mut { $value } { at_value => unsafe { // Do not wrap `$value` in an `unsafe` block.
    //         $crate::pin::Pin::<&mut _>::new_unchecked(at_value)
    //     }}
    // )}
    // ```
    //
    // Safety:
    //   - `type P = &mut _`. There are thus no pathological `Deref{,Mut}` impls
    //     that would break `Pin`'s invariants.
    //   - `{ $value }` is braced, making it a _block expression_, thus **moving**
    //     the given `$value`, and making it _become an **anonymous** temporary_.
    //     By virtue of being anonymous, it can no longer be accessed, thus
    //     preventing any attempts to `mem::replace` it or `mem::forget` it, _etc._
    //
    // This gives us a `pin!` definition that is sound, and which works, but only
    // in certain scenarios:
    //   - If the `pin!(value)` expression is _directly_ fed to a function call:
    //     `let poll = pin!(fut).poll(cx);`
    //   - If the `pin!(value)` expression is part of a scrutinee:
    //     ```rust
    //     match pin!(fut) { pinned_fut => {
    //         pinned_fut.as_mut().poll(...);
    //         pinned_fut.as_mut().poll(...);
    //     }} // <- `fut` is dropped here.
    //     ```
    // Alas, it doesn't work for the more straight-forward use-case: `let` bindings.
    // ```rust
    // let pinned_fut = pin!(fut); // <- temporary value is freed at the end of this statement
    // pinned_fut.poll(...) // error[E0716]: temporary value dropped while borrowed
    //                      // note: consider using a `let` binding to create a longer lived value
    // ```
    //   - Issues such as this one are the ones motivating https://github.com/rust-lang/rfcs/pull/66
    //
    // This makes such a macro incredibly unergonomic in practice, and the reason most macros
    // out there had to take the path of being a statement/binding macro (_e.g._, `pin!(future);`)
    // instead of featuring the more intuitive ergonomics of an expression macro.
    //
    // Luckily, there is a way to avoid the problem. Indeed, the problem stems from the fact that a
    // temporary is dropped at the end of its enclosing statement when it is part of the parameters
    // given to function call, which has precisely been the case with our `Pin::new_unchecked()`!
    // For instance,
    // ```rust
    // let p = Pin::new_unchecked(&mut <temporary>);
    // ```
    // becomes:
    // ```rust
    // let p = { let mut anon = <temporary>; &mut anon };
    // ```
    //
    // However, when using a literal braced struct to construct the value, references to temporaries
    // can then be taken. This makes Rust change the lifespan of such temporaries so that they are,
    // instead, dropped _at the end of the enscoping block_.
    // For instance,
    // ```rust
    // let p = Pin { __pointer: &mut <temporary> };
    // ```
    // becomes:
    // ```rust
    // let mut anon = <temporary>;
    // let p = Pin { __pointer: &mut anon };
    // ```
    // which is *exactly* what we want.
    //
    // See https://doc.rust-lang.org/1.58.1/reference/destructors.html#temporary-lifetime-extension
    // for more info.
    $crate::pin::Pin::<&mut _> { __pointer: &mut { $value } }
}
