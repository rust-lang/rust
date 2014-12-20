- Start Date: 2014-12-20
- RFC PR: https://github.com/rust-lang/rfcs/pull/503
- Rust Issue: https://github.com/rust-lang/rust/issues/20068

# Summary

Stabilize the `std::prelude` module by removing some of the less commonly used
functionality of it.

# Motivation

The prelude of the standard library is included into all Rust programs by
default, and is consequently quite an important module to consider when
stabilizing the standard library. Some of the primary tasks of the prelude are:

* The prelude is used to represent imports that would otherwise occur in nearly
  all Rust modules. The threshold for entering the prelude is consequently quite
  high as it is unlikely to be able to change in a backwards compatible fashion
  as-is.
* Primitive types such as `str` and `char` are unable to have inherent methods
  attached to them. In order to provide methods extension traits must be used.
  All of these traits are members of the prelude in order to enable methods on
  language-defined types.

This RFC currently focuses on removing functionality from the prelude rather
than adding it. New additions can continue to happen before 1.0 and will be
evaluated on a case-by-case basis. The rationale for removal or inclusion will
be provided below.

# Detailed Design

The current `std::prelude` module was copied into the document of this RFC, and
each reexport should be listed below and categorized. The rationale for
inclusion of each type is included inline.

## Reexports to retain

This section provides the exact prelude that this RFC proposes:

```rust
// Boxes are a ubiquitous type in Rust used for representing an allocation with
// a known fixed address. It is also one of the canonical examples of an owned
// type, appearing in many examples and tests. Due to its common usage, the Box
// type is present.
pub use boxed::Box;

// These two traits are present to provide methods on the `char` primitive type.
// The two traits will be collapsed into one `CharExt` trait in the `std::char`
// module, however instead of reexporting two traits.
pub use char::{Char, UnicodeChar};

// One of the most common operations when working with references in Rust is the
// `clone()` method to promote the reference to an owned value. As one of the
// core concepts in Rust used by virtually all programs, this trait is included
// in the prelude.
pub use clone::Clone;

// It is expected that these traits will be used in generic bounds much more
// frequently than there will be manual implementations. This common usage in
// bounds to provide the fundamental ability to compare two values is the reason
// for the inclusion of these traits in the prelude.
pub use cmp::{PartialEq, PartialOrd, Eq, Ord};

// Iterators are one of the most core primitives in the standard libary which is
// used to interoperate between any sort of sequence of data. Due to the
// widespread use, these traits and extension traits are all present in the
// prelude.
//
// The `Iterator*Ext` traits can be removed if generalized where clauses for
// methods are implemented, and they are currently included to represent the
// functionality provided today. The various traits other than `Iterator`, such
// as `DoubleEndedIterator` and `ExactSizeIterator` are provided in order to
// ensure that the methods are available like the `Iterator` methods.
pub use iter::{DoubleEndedIteratorExt, CloneIteratorExt};
pub use iter::{Extend, ExactSizeIterator};
pub use iter::{Iterator, IteratorExt, DoubleEndedIterator};
pub use iter::{IteratorCloneExt};
pub use iter::{IteratorOrdExt};

// As core language concepts and frequently used bounds on generics, these kinds
// are all included in the prelude by default. Note, however, that the exact
// set of kinds in the prelude will be determined by the stabilization of this
// module.
pub use kinds::{Copy, Send, Sized, Sync};

// One of Rust's fundamental principles is ownership, and understanding movement
// of types is key to this. The drop function, while a convenience, represents
// the concept of ownership and relinquishing ownership, so it is included.
pub use mem::drop;

// As described below, very few `ops` traits will continue to remain in the
// prelude. `Drop`, however, stands out from the other operations for many of
// the similar reasons as to the `drop` function.
pub use ops::Drop;

// Similarly to the `cmp` traits, these traits are expected to be bounds on
// generics quite commonly to represent a pending computation that can be
// executed.
pub use ops::{Fn, FnMut, FnOnce};

// The `Option` type is one of Rust's most common and ubiquitous types,
// justifying its inclusion into the prelude along with its two variants.
pub use option::Option::{mod, Some, None};

// In order to provide methods on raw pointers, these two traits are included
// into the prelude. It is expected that these traits will be renamed to
// `PtrExt` and `MutPtrExt`.
pub use ptr::{RawPtr, RawMutPtr};

// This type is included for the same reasons as the `Option` type.
pub use result::Result::{mod, Ok, Err};

// The slice family of traits are all provided in order to export methods on the
// language slice type. The `SlicePrelude` and `SliceAllocPrelude` will be
// collapsed into one `SliceExt` trait by the `std::slice` module. Many of the
// remaining traits require generalized where clauses on methods to be merged
// into the `SliceExt` trait, which may not happen for 1.0.
pub use slice::{SlicePrelude, SliceAllocPrelude, CloneSlicePrelude};
pub use slice::{CloneSliceAllocPrelude, OrdSliceAllocPrelude};
pub use slice::{PartialEqSlicePrelude, OrdSlicePrelude};

// These traits, like the above traits, are providing inherent methods on
// slices, but are not candidates for merging into `SliceExt`. Nevertheless
// these common operations are included for the purpose of adding methods on
// language-defined types.
pub use slice::{BoxedSlicePrelude, AsSlice, VectorVector};

// The str family of traits provide inherent methods on the `str` type. The
// `StrPrelude`, `StrAllocating`, and `UnicodeStrPrelude` traits will all be
// collapsed into one `StrExt` trait to be reexported in the prelude. The `Str`
// trait itself will be handled in the stabilization of the `str` module, but
// for now is included for consistency. Similarly, the `StrVector` trait is
// still undergoing stabilization but remains for consistency.
pub use str::{Str, StrPrelude};
pub use str::{StrAllocating, UnicodeStrPrelude};
pub use str::{StrVector};

// As the standard library's default owned string type, `String` is provided in
// the prelude. Many of the same reasons for `Box`'s inclusion apply to `String`
// as well.
pub use string::String;

// Converting types to a `String` is seen as a common-enough operation for
// including this trait in the prelude.
pub use string::ToString;

// Included for the same reasons as `String` and `Box`.
pub use vec::Vec;
```

## Reexports to remove

All of the following reexports are currently present in the prelude and are
proposed for removal by this RFC.

```rust
// While currently present in the prelude, these traits do not need to be in
// scope to use the language syntax associated with each trait. These traits are
// also only rarely used in bounds on generics and are consequently
// predominately used for `impl` blocks. Due to this lack of need to be included
// into all modules in Rust, these traits are all removed from the prelude.
pub use ops::{Add, Sub, Mul, Div, Rem, Neg, Not};
pub use ops::{BitAnd, BitOr, BitXor};
pub use ops::{Deref, DerefMut};
pub use ops::{Shl, Shr};
pub use ops::{Index, IndexMut};
pub use ops::{Slice, SliceMut};

// Now that tuple indexing is a feature of the language, these traits are no
// longer necessary and can be deprecated.
pub use tuple::{Tuple1, Tuple2, Tuple3, Tuple4};
pub use tuple::{Tuple5, Tuple6, Tuple7, Tuple8};
pub use tuple::{Tuple9, Tuple10, Tuple11, Tuple12};

// Interoperating with ascii data is not necessarily a core language operation
// and the ascii module itself is currently undergoing stabilization. The design
// will likely end up with only one trait (as opposed to the many listed here).
// The prelude will be responsible for providing unicode-respecting methods on
// primitives while requiring that ascii-specific manipulation is imported
// manually.
pub use ascii::{Ascii, AsciiCast, OwnedAsciiCast, AsciiStr};
pub use ascii::IntoBytes;

// Inclusion of this trait is mostly a relic of old behavior and there is very
// little need for the `into_cow` method to be ubiquitously available. Although
// mostly used in bounds on generics, this trait is not itself as commonly used
// as `FnMut`, for example.
pub use borrow::IntoCow;

// The `c_str` module is currently undergoing stabilization as well, but it's
// unlikely for `to_c_str` to be a common operation in almost all Rust code in
// existence, so this trait, if it survives stabilization, is removed from the
// prelude.
pub use c_str::ToCStr;

// This trait is `#[experimental]` in the `std::cmp` module and the prelude is
// intended to be a stable subset of Rust. If later marked #[stable] the trait
// may re-enter the prelude but it will be removed until that time.
pub use cmp::Equiv;

// Actual usage of the `Ordering` enumeration and its variants is quite rare in
// Rust code. Implementors of the `Ord` and `PartialOrd` traits will likely be
// required to import these names, but it is not expected that Rust code at
// large will require these names to be in the prelude.
pub use cmp::Ordering::{mod, Less, Equal, Greater};

// With language-defined `..` syntax there is no longer a need for the `range`
// function to remain in the prelude. This RFC does, however, recommend leaving
// this function in the prelude until the `..` syntax is implemented in order to
// provide a smoother deprecation strategy.
pub use iter::range;

// The FromIterator trait does not need to be present in the prelude as it is
// not adding methods to iterators and is mostly only required to be imported by
// implementors, which is not common enough for inclusion.
pub use iter::{FromIterator};

// Like `cmp::Equiv`, these two iterators are `#[experimental]` and are
// consequently removed from the prelude.
pub use iter::{RandomAccessIterator, MutableDoubleEndedIterator};

// I/O stabilization will have its own RFC soon, and part of that RFC involves
// creating a `std::io::prelude` module which will become the home for these
// traits. This RFC proposes leaving these in the current prelude, however,
// until the I/O stabilization is complete.
pub use io::{Buffer, Writer, Reader, Seek, BufferPrelude};

// These two traits are relics of an older `std::num` module which need not be
// included in the prelude any longer. Their methods are not called often, nor
// are they taken as bounds frequently enough to justify inclusion into the
// prelude.
pub use num::{ToPrimitive, FromPrimitive};

// As part of the Path stabilization RFC, these traits and structures will be
// removed from the prelude. Note that the ergonomics of opening a File today
// will decrease in the sense that `Path` must be imported, but eventually
// importing `Path` will not be necessary due to the `AsPath` trait. More
// details can be found in the path stabilization RFC.
pub use path::{GenericPath, Path, PosixPath, WindowsPath};

// This function is included in the prelude as a convenience function for the
// `FromStr::from_str` associated function. Inclusion of this method, however,
// is inconsistent with respect to the lack of inclusion of a `default` method,
// for example. It is also not necessarily seen as `from_str` being common
// enough to justify its inclusion.
pub use str::from_str;

// This trait is currently only implemented for `Vec<Ascii>` which is likely to
// be removed as part of `std::ascii` stabilization, obsoleting the need for the
// trait and its inclusion in the prelude.
pub use string::IntoString;

// The focus of Rust's story about concurrent program has been constantly
// shifting since it was incepted, and the prelude doesn't necessarily always
// keep up. Message passing is only one form of concurrent primitive that Rust
// provides, and inclusion in the prelude can provide the wrong impression that
// it is the *only* concurrent primitive that Rust offers. In order to
// facilitate a more unified front in Rust's concurrency story, these primitives
// will be removed from the prelude (and soon moved to std::sync as well).
//
// Additionally, while spawning a new thread is a common operation in concurrent
// programming, it is not a frequent operation in code in general. For example
// even highly concurrent applications may end up only calling `spawn` in one or
// two locations which does not necessarily justify its inclusion in the prelude
// for all Rust code in existence.
pub use comm::{sync_channel, channel};
pub use comm::{SyncSender, Sender, Receiver};
pub use task::spawn;
```

## Move to an inner `v1` module

This RFC also proposes moving all reexports to `std::prelude::v1` module instead
of just inside `std::prelude`. The compiler will then start injecting `use
std::prelude::v1::*`.

This is a pre-emptive move to help provide room to grow the prelude module over
time. It is unlikely that any reexports could ever be added to the prelude
backwards-compatibly, so newer preludes (which may happen over time) will have
to live in new modules. If the standard library grows multiple preludes over
time, then it is expected for crates to be able to specify which prelude they
would like to be compiled with. This feature is left as an open question,
however, and movement to an inner `v1` module is simply preparation for this
possible move happening in the future.

The versioning scheme for the prelude over time (if it happens) is also left as
an open question by this RFC.

# Drawbacks

A fairly large amount of functionality was removed from the prelude in order to
hone in on the driving goals of the prelude, but this unfortunately means that
many imports must be added throughout code currently using these reexports. It
is expected, however, that the most painful removals will have roughtly equal
ergonomic replacements in the future. For example:

* Removal of `Path` and friends will retain the current level of ergonomics with
  no imports via the `AsPath` trait.
* Removal of `iter::range` will be replaced via the *more* ergonomic `..`
  syntax.

Many other cases which may be initially seen as painful to migrate are intended
to become aligned with other Rust conventions and practices today. For example
getting into the habit of importing implemented traits (such as the `ops`
traits) is consistent with how many implementations will work. Similarly removal
of synchronization primitives allows for consistence in usage of all concurrent
primitives that Rust provides.

# Alternatives

A number of alternatives were discussed above, and this section can otherwise
largely be filled with various permutations of moving reexports between the
"keep" and "remove" sections above.

# Unresolved Questions

This RFC is fairly aggressive about removing functionality from the prelude, but
is unclear how necessary this is. If Rust grows the ability to
backwards-compatibly modify the prelude in some fasion (for example introducing
multiple preludes that can be opted into) then the aggressive removal may not be
necessary.

If user-defined preludes are allowed in some form, it is also unclear about how
this would impact the inclusion of reexports in the standard library's prelude
in some form.
