// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Both `PLACE <- EXPR` and `box EXPR` desugar into expressions
/// that allocate an intermediate "place" that holds uninitialized
/// state.  The desugaring evaluates EXPR, and writes the result at
/// the address returned by the `pointer` method of this trait.
///
/// A `Place` can be thought of as a special representation for a
/// hypothetical `&uninit` reference (which Rust cannot currently
/// express directly). That is, it represents a pointer to
/// uninitialized storage.
///
/// The client is responsible for two steps: First, initializing the
/// payload (it can access its address via `pointer`). Second,
/// converting the agent to an instance of the owning pointer, via the
/// appropriate `finalize` method (see the `InPlace`.
///
/// If evaluating EXPR fails, then it is up to the destructor for the
/// implementation of Place to clean up any intermediate state
/// (e.g. deallocate box storage, pop a stack, etc).
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait Place<Data: ?Sized> {
    /// Returns the address where the input value will be written.
    /// Note that the data at this address is generally uninitialized,
    /// and thus one should use `ptr::write` for initializing it.
    fn pointer(&mut self) -> *mut Data;
}

/// Interface to implementations of  `PLACE <- EXPR`.
///
/// `PLACE <- EXPR` effectively desugars into:
///
/// ```rust,ignore
/// let p = PLACE;
/// let mut place = Placer::make_place(p);
/// let raw_place = Place::pointer(&mut place);
/// let value = EXPR;
/// unsafe {
///     std::ptr::write(raw_place, value);
///     InPlace::finalize(place)
/// }
/// ```
///
/// The type of `PLACE <- EXPR` is derived from the type of `PLACE`;
/// if the type of `PLACE` is `P`, then the final type of the whole
/// expression is `P::Place::Owner` (see the `InPlace` and `Boxed`
/// traits).
///
/// Values for types implementing this trait usually are transient
/// intermediate values (e.g. the return value of `Vec::emplace_back`)
/// or `Copy`, since the `make_place` method takes `self` by value.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait Placer<Data: ?Sized> {
    /// `Place` is the intermedate agent guarding the
    /// uninitialized state for `Data`.
    type Place: InPlace<Data>;

    /// Creates a fresh place from `self`.
    fn make_place(self) -> Self::Place;
}

/// Specialization of `Place` trait supporting `PLACE <- EXPR`.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait InPlace<Data: ?Sized>: Place<Data> {
    /// `Owner` is the type of the end value of `PLACE <- EXPR`
    ///
    /// Note that when `PLACE <- EXPR` is solely used for
    /// side-effecting an existing data-structure,
    /// e.g. `Vec::emplace_back`, then `Owner` need not carry any
    /// information at all (e.g. it can be the unit type `()` in that
    /// case).
    type Owner;

    /// Converts self into the final value, shifting
    /// deallocation/cleanup responsibilities (if any remain), over to
    /// the returned instance of `Owner` and forgetting self.
    unsafe fn finalize(self) -> Self::Owner;
}

/// Core trait for the `box EXPR` form.
///
/// `box EXPR` effectively desugars into:
///
/// ```rust,ignore
/// let mut place = BoxPlace::make_place();
/// let raw_place = Place::pointer(&mut place);
/// let value = EXPR;
/// unsafe {
///     ::std::ptr::write(raw_place, value);
///     Boxed::finalize(place)
/// }
/// ```
///
/// The type of `box EXPR` is supplied from its surrounding
/// context; in the above expansion, the result type `T` is used
/// to determine which implementation of `Boxed` to use, and that
/// `<T as Boxed>` in turn dictates determines which
/// implementation of `BoxPlace` to use, namely:
/// `<<T as Boxed>::Place as BoxPlace>`.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait Boxed {
    /// The kind of data that is stored in this kind of box.
    type Data;  /* (`Data` unused b/c cannot yet express below bound.) */
    /// The place that will negotiate the storage of the data.
    type Place: BoxPlace<Self::Data>;

    /// Converts filled place into final owning value, shifting
    /// deallocation/cleanup responsibilities (if any remain), over to
    /// returned instance of `Self` and forgetting `filled`.
    unsafe fn finalize(filled: Self::Place) -> Self;
}

/// Specialization of `Place` trait supporting `box EXPR`.
#[unstable(feature = "placement_new_protocol", issue = "27779")]
pub trait BoxPlace<Data: ?Sized> : Place<Data> {
    /// Creates a globally fresh place.
    fn make_place() -> Self;
}
