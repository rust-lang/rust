- Start Date: 2014-09-19
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Remove the reference-counting based `Gc<T>` type from the standard
library and its associated support infrastructure from `rustc`.

Doing so lays a cleaner foundation upon which to prototype a proper
tracing GC, and will avoid people getting incorrect impressions of
Rust based on the current reference-counting implementation.

# Motivation

## Ancient History

Long ago, the Rust language had integrated support for automatically
managed memory with arbitrary graph structure (notably, multiple
references to the same object), via the type constructors `@T` and
`@mut T` for any `T`.  The intention was that Rust would provide a
task-local garbage collector as part of the standard runtime for Rust
programs.

As a short-term convenience, `@T` and `@mut T` were implemented via
reference-counting: each instance of `@T`/`@mut T` had a reference
count added to it (as well as other meta-data that were again for
implementation convenience).  To support this, the `rustc` compiler
would emit, for any instruction copying or overwriting an instance of
`@T`/`@mut T`, code to update the reference count(s) accordingly.

(At the same time, `@T` was still considered an instance of `Copy` by
the compiler.  Maintaining the reference counts of `@T` means that you
*cannot* create copies of a given type implementing `Copy` by
`memcpy`'ing blindly; one must distinguish so-called "POD" data that
is `Copy and contains no `@T` from "non-POD" `Copy` data that can
contain `@T` and thus must be sure to update reference counts when
creating a copy.)

Over time, `@T` was replaced with the library type `Gc<T>` (and `@mut
T` was rewritten as `Gc<RefCell<T>>`), but the intention was that Rust
would still have integrated support for a garbage collection.  To
continue supporting the reference-count updating semantics, the
`Gc<T>` type has a lang item, `"gc"`.  In effect, all of the compiler
support for maintaining the reference-counts from the prior `@T` was
still in place; the move to a library type `Gc<T>` was just a shift in
perspective from the end-user's point of view (and that of the
parser).

## Recent history: Removing uses of Gc<T> from the compiler

Largely due to the tireless efforts of `eddyb`, one of the primary
clients of `Gc<T>`, namely the `rustc` compiler itself, has little to
no remaining uses of `Gc<T>`.

## A new hope

This means that we have an opportunity now, to remove the `Gc<T>` type
from `libstd`, and its associated built-in reference-counting support
from `rustc` itself.

I want to distinguish removal of the particular reference counting
`Gc<T>` from our compiler and standard library (which is what is being
proposed here), from removing the goal of supporting a garbage
collected `Gc<T>` in the future. I (and I think the majority of the
Rust core team) still believe that there are use cases that would be
well handled by a proper tracing garbage collector.

The expected outcome of removing reference-counting `Gc<T>` are as follows:

 * A cleaner compiler code base,

 * A cleaner standard library, where `Copy` data can be indeed copied
    blindly (assuming the source and target types are in agreement,
    which is required for a tracing GC),

 * It would become impossible for users to use `Gc<T>` and then get
   incorrect impressions about how Rust's GC would behave in the
   future.  In particular, if we leave the reference-counting `Gc<T>`
   in place, then users may end up depending on implementation
   artifacts that we would be pressured to continue supporting in the
   future.  (Note that `Gc<T>` is already marked "experimental", so
   this particular motivation is not very strong.)

# Detailed design

Remove the `std::gc` module.  This, I believe, is the extent of the
end-user visible changes proposed by this RFC, at least for users who
are using `libstd` (as opposed to implementing their own).

Then remove the `rustc` support for `Gc<T>`. As part of this, we can
either leave in or remove the `"gc"` and `"managed_heap"` entries in
the lang items table (in case they could be of use for a future GC
implementation).  I propose leaving them, but it does not matter
terribly to me.  The important thing is that once `std::gc` is gone,
then we can remove the support code associated with those two lang
items, which is the important thing.

# Drawbacks

Taking out the reference-counting `Gc<T>` now may lead people to think
that Rust will never have a `Gc<T>`.

 * In particular, having `Gc<T>` in place now means that it is easier
   to argue for putting in a tracing collector (since it would be a
   net win over the status quo, assuming it works).

   (This sub-bullet is a bit of a straw man argument, as I suspect any
   community resistance to adding a tracing GC will probably be
   unaffected by the presence or absence of the reference-counting
   `Gc<T>`.)

 * As another related note, it may confuse people to take out a
   `Gc<T>` type now only to add another implementation with the same
   name later.  (Of course, is that more or less confusing than just
   replacing the underlying implementation in such a severe manner.)

Users may be using `Gc<T>` today, and they would have to switch to
some other option (such as `Rc<T>`, though note that the two are not
100% equivalent).

# Alternatives

Keep the `Gc<T>` implementation that we have today, and wait until we
have a tracing GC implemented and ready to be deployed before removing
the reference-counting infrastructure that had been put in to support
`@T`.  (Which may never happen, since adding a tracing GC is only a
goal, not a certainty, and thus we may be stuck supporting the
reference-counting `Gc<T>` until we eventually do decide to remove
`Gc<T>` in the future.  So this RFC is just suggesting we be proactive
and pull that band-aid off now.

# Unresolved questions

None yet.
