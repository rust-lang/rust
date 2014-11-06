// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# Method lookup

Method lookup can be rather complex due to the interaction of a number
of factors, such as self types, autoderef, trait lookup, etc. This
file provides an overview of the process. More detailed notes are in
the code itself, naturally.

One way to think of method lookup is that we convert an expression of
the form:

    receiver.method(...)

into a more explicit UFCS form:

    Trait::method(ADJ(receiver), ...) // for a trait call
    ReceiverType::method(ADJ(receiver), ...) // for an inherent method call

Here `ADJ` is some kind of adjustment, which is typically a series of
autoderefs and then possibly an autoref (e.g., `&**receiver`). However
we sometimes do other adjustments and coercions along the way, in
particular unsizing (e.g., converting from `[T, ..n]` to `[T]`).

## The Two Phases

Method lookup is divided into two major phases: probing (`probe.rs`)
and confirmation (`confirm.rs`). The probe phase is when we decide
what method to call and how to adjust the receiver. The confirmation
phase "applies" this selection, updating the side-tables, unifying
type variables, and otherwise doing side-effectful things.

One reason for this division is to be more amenable to caching.  The
probe phase produces a "pick" (`probe::Pick`), which is designed to be
cacheable across method-call sites. Therefore, it does not include
inference variables or other information.

## Probe phase

The probe phase (`probe.rs`) decides what method is being called and
how to adjust the receiver.

### Steps

The first thing that the probe phase does is to create a series of
*steps*. This is done by progressively dereferencing the receiver type
until it cannot be deref'd anymore, as well as applying an optional
"unsize" step. So if the receiver has type `Rc<Box<[T, ..3]>>`, this
might yield:

    Rc<Box<[T, ..3]>>
    Box<[T, ..3]>
    [T, ..3]
    [T]

### Candidate assembly

We then search along those steps to create a list of *candidates*. A
`Candidate` is a method item that might plausibly be the method being
invoked. For each candidate, we'll derive a "transformed self type"
that takes into account explicit self.

Candidates are grouped into two kinds, inherent and extension.

**Inherent candidates** are those that are derived from the
type of the receiver itself.  So, if you have a receiver of some
nominal type `Foo` (e.g., a struct), any methods defined within an
impl like `impl Foo` are inherent methods.  Nothing needs to be
imported to use an inherent method, they are associated with the type
itself (note that inherent impls can only be defined in the same
module as the type itself).

FIXME: Inherent candidates are not always derived from impls.  If you
have a trait object, such as a value of type `Box<ToString>`, then the
trait methods (`to_string()`, in this case) are inherently associated
with it. Another case is type parameters, in which case the methods of
their bounds are inherent. However, this part of the rules is subject
to change: when DST's "impl Trait for Trait" is complete, trait object
dispatch could be subsumed into trait matching, and the type parameter
behavior should be reconsidered in light of where clauses.

**Extension candidates** are derived from imported traits.  If I have
the trait `ToString` imported, and I call `to_string()` on a value of
type `T`, then we will go off to find out whether there is an impl of
`ToString` for `T`.  These kinds of method calls are called "extension
methods".  They can be defined in any module, not only the one that
defined `T`.  Furthermore, you must import the trait to call such a
method.

So, let's continue our example. Imagine that we were calling a method
`foo` with the receiver `Rc<Box<[T, ..3]>>` and there is a trait `Foo`
that defines it with `&self` for the type `Rc<U>` as well as a method
on the type `Box` that defines `Foo` but with `&mut self`. Then we
might have two candidates:

    &Rc<Box<[T, ..3]>> from the impl of `Foo` for `Rc<U>` where `U=Box<T, ..3]>
    &mut Box<[T, ..3]>> from the inherent impl on `Box<U>` where `U=[T, ..3]`

### Candidate search

Finally, to actually pick the method, we will search down the steps,
trying to match the receiver type against the candidate types. At
each step, we also consider an auto-ref and auto-mut-ref to see whether
that makes any of the candidates match. We pick the first step where
we find a match.

In the case of our example, the first step is `Rc<Box<[T, ..3]>>`,
which does not itself match any candidate. But when we autoref it, we
get the type `&Rc<Box<[T, ..3]>>` which does match. We would then
recursively consider all where-clauses that appear on the impl: if
those match (or we cannot rule out that they do), then this is the
method we would pick. Otherwise, we would continue down the series of
steps.

*/

