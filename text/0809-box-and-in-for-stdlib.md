- Feature Name: box_syntax, placement_in_syntax
- Start Date: 2015-02-04
- RFC PR: [rust-lang/rfcs#809](https://github.com/rust-lang/rfcs/pull/809)
- Rust Issue: [rust-lang/rust#22181](https://github.com/rust-lang/rust/issues/22181)

# Summary

  * Change placement-new syntax from: `box (<place-expr>) <expr>` instead
    to: `in <place-expr> { <block> }`.

  * Change `box <expr>` to an overloaded operator that chooses its
    implementation based on the expected type.

  * Use unstable traits in `core::ops` for both operators, so that
    libstd can provide support for the overloaded operators; the
    traits are unstable so that the language designers are free to
    revise the underlying protocol in the future post 1.0.

  * Feature-gate the placement-`in` syntax via the feature name `placement_in_syntax`.

  * The overloaded `box <expr>` will reuse the `box_syntax` feature name.

(Note that `<block>` here denotes the interior of a block expression; i.e.:
```
<block> ::= [ <stmt> ';' | <item> ] * [ <expr> ]
```
This is the same sense in which the `block` nonterminal is used in the
reference manual.)

# Motivation

Goal 1: We want to support an operation analogous to C++'s placement
new, as discussed previously in [Placement Box RFC PR 470].

[Placement Box RFC PR 470]: https://github.com/rust-lang/rfcs/pull/470

Goal 2: We also would like to overload our `box` syntax so that more
types, such as `Rc<T>` and `Arc<T>` can gain the benefit of avoiding
intermediate copies (i.e. allowing expressions to install their result
value directly into the backing storage of the `Rc<T>` or `Arc<T>`
when it is created).

However, during discussion of [Placement Box RFC PR 470], some things
became clear:

 *  Many syntaxes using the `in` keyword are superior to `box (<place-expr>)
    <expr>` for the operation analogous to placement-new.

    The proposed `in`-based syntax avoids ambiguities such as having
    to write `box () (<expr>)` (or `box (alloc::HEAP) (<expr>)`) when
    one wants to surround `<expr>` with parentheses.
    It allows the parser to provide clearer error messages when
    encountering `in <place-expr> <expr>` (clearer compared to the previous
    situation with `box <place-expr> <expr>`).

 *  It would be premature for Rust to commit to any particular
    protocol for supporting placement-`in`. A number of participants in
    the discussion of [Placement Box RFC PR 470] were unhappy with the
    baroque protocol, especially since it did not support DST and
    potential future language changes would allow the protocol
    proposed there to be significantly simplified.

Therefore, this RFC proposes a middle ground for 1.0: Support the
desired syntax, but do not provide stable support for end-user
implementations of the operators. The only stable ways to use the
overloaded `box <expr>` or `in <place-expr> { <block> }` operators will be in
tandem with types provided by the stdlib, such as `Box<T>`.

# Detailed design

* Add traits to `core::ops` for supporting the new operators.
  This RFC does not commit to any particular set of traits,
  since they are not currently meant to be implemented outside
  of the stdlib. (However, a demonstration of one working set
  of traits is given in [Appendix A].)

  Any protocol that we adopt for the operators needs to properly
  handle panics; i.e., `box <expr>` must properly cleanup any
  intermediate state if `<expr>` panics during its evaluation,
  and likewise for `in <place-expr> { <block> }`

  (See [Placement Box RFC PR 470] or [Appendix A] for discussion on
   ways to accomplish this.)

* Change `box <expr>` from built-in syntax (tightly integrated with
  `Box<T>`) into an overloaded-`box` operator that uses the expected
  return type to decide what kind of value to create.  For example, if
  `Rc<T>` is extended with an implementation of the appropriate
  operator trait, then

  ```rust
  let x: Rc<_> = box format!("Hello");
  ```

  could be a legal way to create an `Rc<String>` without having to
  invoke the `Rc::new` function. This will be more efficient for
  building instances of `Rc<T>` when `T` is a large type.  (It is also
  arguably much cleaner syntax to read, regardless of the type `T`.)

  Note that this change will require end-user code to no longer assume
  that `box <expr>` always produces a `Box<T>`; such code will need to
  either add a type annotation e.g. saying `Box<_>`, or will need to
  call `Box::new(<expr>)` instead of using `box <expr>`.

* Add support for parsing `in <place-expr> { <block> }` as the basis for the
  placement operator.

  Remove support for `box (<place-expr>) <expr>` from the parser.

  Make `in <place-expr> { <block> }` an overloaded operator that uses
  the `<place-expr>` to determine what placement code to run.

  Note: when `<place-expr>` is just an identifier,
  `<place-expr> { <block> }` is not parsed as a struct literal.
  We accomplish this via the same means that is used e.g. for `if` expressions:
  we restrict `<place-expr>` to not include struct literals
  (see [RFC 92]).

[RFC 92]: https://github.com/rust-lang/rfcs/blob/master/text/0092-struct-grammar.md

* The only stablized implementation for the `box <expr>` operator
  proposed by this RFC is `Box<T>`. The question of which other types
  should support integration with `box <expr>` is a library design
  issue and needs to go through the conventions and library
  stabilization process.

  Similarly, this RFC does not propose *any* stablized implementation
  for the `in <place-expr> { <block> }` operator. (An obvious candidate for
  `in <place-expr> { <block> }` integration would be a `Vec::emplace_back`
  method; but again, the choice of which such methods to add is a
  library design issue, beyond the scope of this RFC.)

  (A sample implementation illustrating how to support the operators
  on other types is given in [Appendix A].)

* Feature-gate the two syntaxes under separate feature identifiers, so that we
  have the option of removing the gate for one syntax without the other.
  (I.e. we already have much experience with non-overloaded `box <expr>`,
   but we have nearly no experience with placement-`in` as described here).

# Drawbacks

* End-users might be annoyed that they cannot add implementations of
  the overloaded-`box` and placement-`in` operators themselves. But
  such users who want to do such a thing will probably be using the
  nightly release channel, which will not have the same stability
  restrictions.

* The currently-implemented desugaring does not infer that in an
  expression like `box <expr> as Box<Trait>`, the use of `box <expr>`
  should evaluate to some `Box<_>`. pnkfelix has found that this is
  due to a weakness in compiler itself ([Rust PR 22012]).

  Likewise, the currently-implemented desugaring does not interact
  well with the combination of type-inference and implicit coercions
  to trait objects. That is, when `box <expr>` is used in a context
  like this:
  ```
  fn foo(Box<SomeTrait>) { ... }
  foo(box some_expr());
  ```
  the type inference system attempts to unify the type `Box<SomeTrait>`
  with the return-type of `::protocol::Boxed::finalize(place)`.
  This may also be due to weakness in the compiler, but that is not
  immediately obvious.

  [Appendix B] has a complete code snippet (using a desugaring much like
  the one found in the other appendix) that illustrates two cases of
  interest where this weakness arises.

[Rust PR 22012]: https://github.com/rust-lang/rust/pull/22012

# Alternatives

* We could keep the `box (<place-expr>) <expr>` syntax. It is hard
  to see what the advantage of that is, unless (1.) we can identify
  many cases of types that benefit from supporting both
  overloaded-`box` and placement-`in`, or unless (2.) we anticipate
  some integration with `box` pattern syntax that would motivate using
  the `box` keyword for placement.

* We could use the `in (<place-expr>) <expr>` syntax. An earlier
  version of this RFC used this alternative. It is easier to implement
  on the current code base, but I do not know of any other benefits.
  (Well, maybe parentheses are less "heavyweight" than curly-braces?)

* A number of other syntaxes for placement have been proposed in the
  past; see for example discussion on [RFC PR 405] as well as
  [the previous placement RFC][RFC Surface Syntax Discussion].

  The main constraints I want to meet are:
  1. Do not introduce ambiguity into the grammar for Rust
  2. Maintain left-to-right evaluation order (so the place should
     appear to the left of the value expression in the text).

  But otherwise I am not particularly attached to any single
  syntax.

  One particular alternative that might placate those who object
  to placement-`in`'s `box`-free form would be:
  `box (in <place-expr>) <expr>`.

[RFC PR 405]: https://github.com/rust-lang/rfcs/issues/405

[RFC Surface Syntax Discussion]: https://github.com/pnkfelix/rfcs/blob/fsk-placement-box-rfc/text/0000-placement-box.md#same-semantics-but-different-surface-syntax

* Do nothing. I.e. do not even accept an unstable libstd-only protocol
  for placement-`in` and overloaded-`box`. This would be okay, but
  unfortunate, since in the past some users have identified
  intermediate copies to be a source of inefficiency, and proper use
  of `box <expr>` and placement-`in` can help remove intermediate
  copies.

# Unresolved questions

This RFC represents the current plan for `box`/`in`. However, in the
[RFC discussion][809] a number of questions arose, including possible
design alternatives that might render the `in` keyword unnecessary.
Before the work in this RFC can be unfeature-gated, these questions should
be satisfactorily resolved:

* Can the type-inference and coercion system of the compiler be
  enriched to the point where overloaded `box` and `in` are
  seamlessly usable? Or are type-ascriptions unavoidable when
  supporting overloading?

  In particular, I am assuming here that some amount of current
  weakness cannot be blamed on any particular details of the
  sample desugaring.

  (See [Appendix B] for example code showing weaknesses in
   `rustc` of today.)
* Do we want to change the syntax for `in(place) expr` / `in place { expr }`?
* Do we need `in` at all, or can we replace it with some future possible feature such as `DerefSet` or `&out` etc?
* Do we want to improve the protocol in some way?
  - Note that the protocol was specifically excluded from this RFC.
  - Support for DST expressions such as `box [22, ..count]` (where `count` is a dynamic value)?
  - Protocol making use of more advanced language features?

# Appendices

## Appendix A: sample operator traits
[Appendix A]: #appendix-a-sample-operator-traits

The goal is to show that code like the following can be made to work
in Rust today via appropriate desugarings and trait definitions.

```rust
fn main() {
    use std::rc::Rc;

    let mut v = vec![1,2];
    in v.emplace_back() { 3 }; // has return type `()`
    println!("v: {:?}", v); // prints [1,2,3]

    let b4: Box<i32> = box 4;
    println!("b4: {}", b4);

    let b5: Rc<i32> = box 5;
    println!("b5: {}", b5);

    let b6 = in HEAP { 6 }; // return type Box<i32>
    println!("b6: {}", b6);
}
```

To demonstrate the above, this appendix provides code that runs today;
it demonstrates sample protocols for the proposed operators.
(The entire code-block below should work when e.g. cut-and-paste into
http::play.rust-lang.org )

```rust
#![feature(unsafe_destructor)] // (hopefully unnecessary soon with RFC PR 769)
#![feature(alloc)]

// The easiest way to illustrate the desugaring is by implementing
// it with macros.  So, we will use the macro `in_` for placement-`in`
// and the macro `box_` for overloaded-`box`; you should read
// `in_!( (<place-expr>) <expr> )` as if it were `in <place-expr> { <expr> }`
// and
// `box_!( <expr> )` as if it were `box <expr>`.

// The two macros have been designed to both 1. work with current Rust
// syntax (which in some cases meant avoiding certain associated-item
// syntax that currently causes the compiler to ICE) and 2. infer the
// appropriate code to run based only on either `<place-expr>` (for
// placement-`in`) or on the expected result type (for
// overloaded-`box`).

macro_rules! in_ {
    (($placer:expr) $value:expr) => { {
        let p = $placer;
        let mut place = ::protocol::Placer::make_place(p);
        let raw_place = ::protocol::Place::pointer(&mut place);
        let value = $value;
        unsafe {
            ::std::ptr::write(raw_place, value);
            ::protocol::InPlace::finalize(place)
        }
    } }
}

macro_rules! box_ {
    ($value:expr) => { {
        let mut place = ::protocol::BoxPlace::make_place();
        let raw_place = ::protocol::Place::pointer(&mut place);
        let value = $value;
        unsafe {
            ::std::ptr::write(raw_place, value);
            ::protocol::Boxed::finalize(place)
        }
    } }
}

// Note that while both desugarings are very similar, there are some
// slight differences.  In particular, the placement-`in` desugaring
// uses `InPlace::finalize(place)`, which is a `finalize` method that
// is overloaded based on the `place` argument (the type of which is
// derived from the `<place-expr>` input); on the other hand, the
// overloaded-`box` desugaring uses `Boxed::finalize(place)`, which is
// a `finalize` method that is overloaded based on the expected return
// type. Thus, the determination of which `finalize` method to call is
// derived from different sources in the two desugarings.

// The above desugarings refer to traits in a `protocol` module; these
// are the traits that would be put into `std::ops`, and are given
// below.

mod protocol {

/// Both `in PLACE { BLOCK }` and `box EXPR` desugar into expressions
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
/// If evaluating EXPR fails, then the destructor for the
/// implementation of Place to clean up any intermediate state
/// (e.g. deallocate box storage, pop a stack, etc).
pub trait Place<Data: ?Sized> {
    /// Returns the address where the input value will be written.
    /// Note that the data at this address is generally uninitialized,
    /// and thus one should use `ptr::write` for initializing it.
    fn pointer(&mut self) -> *mut Data;
}

/// Interface to implementations of  `in PLACE { BLOCK }`.
///
/// `in PLACE { BLOCK }` effectively desugars into:
///
/// ```
/// let p = PLACE;
/// let mut place = Placer::make_place(p);
/// let raw_place = Place::pointer(&mut place);
/// let value = { BLOCK };
/// unsafe {
///     std::ptr::write(raw_place, value);
///     InPlace::finalize(place)
/// }
/// ```
///
/// The type of `in PLACE { BLOCK }` is derived from the type of `PLACE`;
/// if the type of `PLACE` is `P`, then the final type of the whole
/// expression is `P::Place::Owner` (see the `InPlace` and `Boxed`
/// traits).
///
/// Values for types implementing this trait usually are transient
/// intermediate values (e.g. the return value of `Vec::emplace_back`)
/// or `Copy`, since the `make_place` method takes `self` by value.
pub trait Placer<Data: ?Sized> {
    /// `Place` is the intermedate agent guarding the
    /// uninitialized state for `Data`.
    type Place: InPlace<Data>;

    /// Creates a fresh place from `self`.
    fn make_place(self) -> Self::Place;
}

/// Specialization of `Place` trait supporting `in PLACE { BLOCK }`.
pub trait InPlace<Data: ?Sized>: Place<Data> {
    /// `Owner` is the type of the end value of `in PLACE { BLOCK }`
    ///
    /// Note that when `in PLACE { BLOCK }` is solely used for
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
/// ```
/// let mut place = BoxPlace::make_place();
/// let raw_place = Place::pointer(&mut place);
/// let value = $value;
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
pub trait Boxed {
    /// The kind of data that is stored in this kind of box.
    type Data;  /* (`Data` unused b/c cannot yet express below bound.) */
    type Place; /* should be bounded by BoxPlace<Self::Data> */

    /// Converts filled place into final owning value, shifting
    /// deallocation/cleanup responsibilities (if any remain), over to
    /// returned instance of `Self` and forgetting `filled`.
    unsafe fn finalize(filled: Self::Place) -> Self;
}

/// Specialization of `Place` trait supporting `box EXPR`.
pub trait BoxPlace<Data: ?Sized> : Place<Data> {
    /// Creates a globally fresh place.
    fn make_place() -> Self;
}

} // end of `mod protocol`

// Next, we need to see sample implementations of these traits.
// First, `Box<T>` needs to support overloaded-`box`: (Note that this
// is not the desired end implementation; e.g.  the `BoxPlace`
// representation here is less efficient than it could be. This is
// just meant to illustrate that an implementation *can* be made;
// i.e. that the overloading *works*.)
//
// Also, just for kicks, I am throwing in `in HEAP { <block> }` support,
// though I do not think that needs to be part of the stable libstd.

struct HEAP;

mod impl_box_for_box {
    use protocol as proto;
    use std::mem;
    use super::HEAP;

    struct BoxPlace<T> { fake_box: Option<Box<T>> }

    fn make_place<T>() -> BoxPlace<T> {
        let t: T = unsafe { mem::zeroed() };
        BoxPlace { fake_box: Some(Box::new(t)) }
    }

    unsafe fn finalize<T>(mut filled: BoxPlace<T>) -> Box<T> {
        let mut ret = None;
        mem::swap(&mut filled.fake_box, &mut ret);
        ret.unwrap()
    }

    impl<'a, T> proto::Placer<T> for HEAP {
        type Place = BoxPlace<T>;
        fn make_place(self) -> BoxPlace<T> { make_place() }
    }

    impl<T> proto::Place<T> for BoxPlace<T> {
        fn pointer(&mut self) -> *mut T {
            match self.fake_box {
                Some(ref mut b) => &mut **b as *mut T,
                None => panic!("impossible"),
            }
        }
    }

    impl<T> proto::BoxPlace<T> for BoxPlace<T> {
        fn make_place() -> BoxPlace<T> { make_place() }
    }

    impl<T> proto::InPlace<T> for BoxPlace<T> {
        type Owner = Box<T>;
        unsafe fn finalize(self) -> Box<T> { finalize(self) }
    }

    impl<T> proto::Boxed for Box<T> {
        type Data = T;
        type Place = BoxPlace<T>;
        unsafe fn finalize(filled: BoxPlace<T>) -> Self { finalize(filled) }
    }
}

// Second, it might be nice if `Rc<T>` supported overloaded-`box`.
//
// (Note again that this may not be the most efficient implementation;
// it is just meant to illustrate that an implementation *can* be
// made; i.e. that the overloading *works*.)
 
mod impl_box_for_rc {
    use protocol as proto;
    use std::mem;
    use std::rc::{self, Rc};

    struct RcPlace<T> { fake_box: Option<Rc<T>> }

    impl<T> proto::Place<T> for RcPlace<T> {
        fn pointer(&mut self) -> *mut T {
            if let Some(ref mut b) = self.fake_box {
                if let Some(r) = rc::get_mut(b) {
                    return r as *mut T
                }
            }
            panic!("impossible");
        }
    }

    impl<T> proto::BoxPlace<T> for RcPlace<T> {
        fn make_place() -> RcPlace<T> {
            unsafe {
                let t: T = mem::zeroed();
                RcPlace { fake_box: Some(Rc::new(t)) }
            }
        }
    }

    impl<T> proto::Boxed for Rc<T> {
        type Data = T;
        type Place = RcPlace<T>;
        unsafe fn finalize(mut filled: RcPlace<T>) -> Self {
            let mut ret = None;
            mem::swap(&mut filled.fake_box, &mut ret);
            ret.unwrap()
        }
    }
}

// Third, we want something to demonstrate placement-`in`. Let us use
// `Vec::emplace_back` for that:

mod impl_in_for_vec_emplace_back {
    use protocol as proto;

    use std::mem;

    struct VecPlacer<'a, T:'a> { v: &'a mut Vec<T> }
    struct VecPlace<'a, T:'a> { v: &'a mut Vec<T> }

    pub trait EmplaceBack<T> { fn emplace_back(&mut self) -> VecPlacer<T>; }

    impl<T> EmplaceBack<T> for Vec<T> {
        fn emplace_back(&mut self) -> VecPlacer<T> { VecPlacer { v: self } }
    }

    impl<'a, T> proto::Placer<T> for VecPlacer<'a, T> {
        type Place = VecPlace<'a, T>;
        fn make_place(self) -> VecPlace<'a, T> { VecPlace { v: self.v } }
    }

    impl<'a, T> proto::Place<T> for VecPlace<'a, T> {
        fn pointer(&mut self) -> *mut T {
            unsafe {
                let idx = self.v.len();
                self.v.push(mem::zeroed());
                &mut self.v[idx]
            }
        }
    }
    impl<'a, T> proto::InPlace<T> for VecPlace<'a, T> {
        type Owner = ();
        unsafe fn finalize(self) -> () {
            mem::forget(self);
        }
    }

    #[unsafe_destructor]
    impl<'a, T> Drop for VecPlace<'a, T> {
        fn drop(&mut self) {
            unsafe {
                mem::forget(self.v.pop())
            }
        }
    }
}

// Okay, that's enough for us to actually demonstrate the syntax!
// Here's our `fn main`:

fn main() {
    use std::rc::Rc;
    // get hacked-in `emplace_back` into scope
    use impl_in_for_vec_emplace_back::EmplaceBack;

    let mut v = vec![1,2];
    in_!( (v.emplace_back()) 3 );
    println!("v: {:?}", v);

    let b4: Box<i32> = box_!( 4 );
    println!("b4: {}", b4);

    let b5: Rc<i32> = box_!( 5 );
    println!("b5: {}", b5);

    let b6 = in_!( (HEAP) 6 ); // return type Box<i32>
    println!("b6: {}", b6);
}
```

## Appendix B: examples of interaction between desugaring, type-inference, and coercion
[Appendix B]: #appendix-b-examples-of-interaction-between-desugaring-type-inference-and-coercion

The following code works with the current version of `box` syntax in Rust, but needs some sort
of type annotation in Rust as it stands today for the desugaring of `box` to work out.

(The following code uses `cfg` attributes to make it easy to switch between slight variations
on the portions that expose the weakness.)

```rust
#![feature(box_syntax)]

// NOTE: Scroll down to "START HERE"

fn main() { }

macro_rules! box_ {
    ($value:expr) => { {
        let mut place = ::BoxPlace::make();
        let raw_place = ::Place::pointer(&mut place);
        let value = $value;
        unsafe { ::std::ptr::write(raw_place, value); ::Boxed::fin(place) }
    } }
}

// (Support traits and impls for examples below.)

pub trait BoxPlace<Data: ?Sized> : Place<Data> { fn make() -> Self; }
pub trait Place<Data: ?Sized> { fn pointer(&mut self) -> *mut Data; }
pub trait Boxed { type Place; fn fin(filled: Self::Place) -> Self; }

struct BP<T: ?Sized> { _fake_box: Option<Box<T>> }

impl<T> BoxPlace<T> for BP<T> { fn make() -> BP<T> { make_pl() } }
impl<T: ?Sized> Place<T> for BP<T> { fn pointer(&mut self) -> *mut T { pointer(self) } }
impl<T: ?Sized> Boxed for Box<T> { type Place = BP<T>; fn fin(x: BP<T>) -> Self { finaliz(x) } }

fn make_pl<T>() -> BP<T> { loop { } }
fn finaliz<T: ?Sized>(mut _filled: BP<T>) -> Box<T> { loop { } }
fn pointer<T: ?Sized>(_p: &mut BP<T>) -> *mut T { loop { } }

// START HERE

pub type BoxFn<'a> = Box<Fn() + 'a>;

#[cfg(all(not(coerce_works1),not(coerce_works2),not(coerce_works3)))]
pub fn coerce<'a, F>(f: F) -> BoxFn<'a> where F: Fn(), F: 'a { box_!( f ) }

#[cfg(coerce_works1)]
pub fn coerce<'a, F>(f: F) -> BoxFn<'a> where F: Fn(), F: 'a {   box  f   }

#[cfg(coerce_works2)]
pub fn coerce<'a, F>(f: F) -> BoxFn<'a> where F: Fn(), F: 'a { let b: Box<_> = box_!( f ); b }

#[cfg(coerce_works3)] // (This one assumes PR 22012 has landed)
pub fn coerce<'a, F>(f: F) -> BoxFn<'a> where F: Fn(), F: 'a { box_!( f ) as BoxFn }


trait Duh { fn duh() -> Self; }

#[cfg(all(not(duh_works1),not(duh_works2)))]
impl<T> Duh for Box<[T]> { fn duh() -> Box<[T]> { box_!( [] ) } }

#[cfg(duh_works1)]
impl<T> Duh for Box<[T]> { fn duh() -> Box<[T]> {   box  [] } }

#[cfg(duh_works2)]
impl<T> Duh for Box<[T]> { fn duh() -> Box<[T]> { let b: Box<[_; 0]> =  box_!( [] ); b } }
```

You can pass `--cfg duh_worksN` and `--cfg coerce_worksM` for suitable
`N` and `M` to see them compile.  Here is a transcript with those attempts,
including the cases where type-inference fails in the desugaring.

```
% rustc /tmp/foo6.rs --cfg duh_works1 --cfg coerce_works1
% rustc /tmp/foo6.rs --cfg duh_works1 --cfg coerce_works2
% rustc /tmp/foo6.rs --cfg duh_works2 --cfg coerce_works1
% rustc /tmp/foo6.rs --cfg duh_works1
/tmp/foo6.rs:10:25: 10:41 error: the trait `Place<F>` is not implemented for the type `BP<core::ops::Fn()>` [E0277]
/tmp/foo6.rs:10         let raw_place = ::Place::pointer(&mut place);
                                        ^~~~~~~~~~~~~~~~
/tmp/foo6.rs:7:1: 14:2 note: in expansion of box_!
/tmp/foo6.rs:37:64: 37:76 note: expansion site
/tmp/foo6.rs:9:25: 9:41 error: the trait `core::marker::Sized` is not implemented for the type `core::ops::Fn()` [E0277]
/tmp/foo6.rs:9         let mut place = ::BoxPlace::make();
                                       ^~~~~~~~~~~~~~~~
/tmp/foo6.rs:7:1: 14:2 note: in expansion of box_!
/tmp/foo6.rs:37:64: 37:76 note: expansion site
error: aborting due to 2 previous errors
% rustc /tmp/foo6.rs                  --cfg coerce_works1
/tmp/foo6.rs:10:25: 10:41 error: the trait `Place<[_; 0]>` is not implemented for the type `BP<[T]>` [E0277]
/tmp/foo6.rs:10         let raw_place = ::Place::pointer(&mut place);
                                        ^~~~~~~~~~~~~~~~
/tmp/foo6.rs:7:1: 14:2 note: in expansion of box_!
/tmp/foo6.rs:52:51: 52:64 note: expansion site
/tmp/foo6.rs:9:25: 9:41 error: the trait `core::marker::Sized` is not implemented for the type `[T]` [E0277]
/tmp/foo6.rs:9         let mut place = ::BoxPlace::make();
                                       ^~~~~~~~~~~~~~~~
/tmp/foo6.rs:7:1: 14:2 note: in expansion of box_!
/tmp/foo6.rs:52:51: 52:64 note: expansion site
error: aborting due to 2 previous errors
% 
```

The point I want to get across is
this: It looks like both of these cases can be worked around via
explicit type ascription.  Whether or not this is an acceptable cost
is a reasonable question.

 * Note that type ascription is especially annoying for the `fn duh` case,
   where one needs to keep the array-length encoded in the type consistent
   with the length of the array generated by the expression.
   This might motivate extending the use of wildcard `_` within type expressions
   to include wildcard constants, for use in the array length, i.e.: `[T; _]`.
   
The `fn coerce` example comes from uses of the `fn combine_structure` function in the
`libsyntax` crate.

The `fn duh` example comes from the implementation of the `Default`
trait for `Box<[T]>`.

Both examples are instances of coercion; the `fn coerce` example is
trying to express a coercion of a `Box<Type>` to a `Box<Trait>`
(i.e. making a trait-object), and the `fn duh` example is trying to
express a coercion of a `Box<[T; k]>` (specifically `[T; 0]`) to a
`Box<[T]>`.  Both are going from a pointer-to-sized to a
pointer-to-unsized.

(Maybe there is a way to handle both of these cases in a generic
fashion; pnkfelix is not sufficiently familiar with how coercions
currently interact with type-inference in the first place.)

[809]: https://github.com/rust-lang/rfcs/pull/809
