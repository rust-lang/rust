- Feature Name: N/A
- Start Date: 2015-04-15
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Alter the signature of the `std::mem::forget` function to remove `unsafe`.
Explicitly state that it is not considered unsafe behavior to not run
destructors.

# Motivation

It was [recently discovered][scoped-bug] by @arielb1 that the `thread::scoped`
API was unsound. To recap, this API previously allowed spawning a child thread
sharing the parent's stack, returning an RAII guard which `join`'d the child
thread when it fell out of scope. The join-on-drop behavior here is critical to
the safety of the API to ensure that the parent does not pop the stack frames
the child is referencing. Put another way, the safety of `thread::scoped` relied
on the fact that the `Drop` implementation for `JoinGuard` was *always* run.

[scoped-bug]: https://github.com/rust-lang/rust/issues/24292

The [underlying issue][forget-bug] for this safety hole was that it is possible
to write a version of `mem::forget` without using `unsafe` code (which drops a
value without running its destructor). This is done by creating a cycle of `Rc`
pointers, leaking the actual contents. It [has been pointed out][dtor-comment]
that `Rc` is not the only vector of leaking contents today as there are
[known][dtor-bug1] [bugs][dtor-bug2] where `panic!` may fail to run
destructors. Furthermore, it has [also been pointed out][drain-bug] that not
running destructors can affect the safety of APIs like `Vec::drain_range` in
addition to `thread::scoped`.

[forget-bug]: https://github.com/rust-lang/rust/issues/24456
[dtor-comment]: https://github.com/rust-lang/rust/issues/24292#issuecomment-93505374
[dtor-bug1]: https://github.com/rust-lang/rust/issues/14875
[dtor-bug2]: https://github.com/rust-lang/rust/issues/16135
[drain-bug]: https://github.com/rust-lang/rust/issues/24292#issuecomment-93513451

It has never been a guarantee of Rust that destructors for a type will run, and
this aspect was overlooked with the `thread::scoped` API which requires that its
destructor be run! Reconciling these two desires has lead to a good deal of
discussion of possible mitigation strategies for various aspects of this
problem. This strategy proposed in this RFC aims to fit uninvasively into the
standard library to avoid large overhauls or destabilizations of APIs.

# Detailed design

Primarily, the `unsafe` annotation on the `mem::forget` function will be
removed, allowing it to be called from safe Rust. This transition will be made
possible by stating that destructors **may not run** in all circumstances (from
both the language and library level). The standard library and the primitives it
provides will always attempt to run destructors, but will not provide a
guarantee that destructors will be run.

It is still likely to be a footgun to call `mem::forget` as memory leaks are
almost always undesirable, but the purpose of the `unsafe` keyword in Rust is to
indicate **memory unsafety** instead of being a general deterrent for "should be
avoided" APIs. Given the premise that types must be written assuming that their
destructor may not run, it is the fault of the type in question if `mem::forget`
would trigger memory unsafety, hence allowing `mem::forget` to be a safe
function.

Note that this modification to `mem::forget` is a breaking change due to the
signature of the function being altered, but it is expected that most code will
not break in practice and this would be an acceptable change to cherry-pick into
the 1.0 release.

# Drawbacks

It is clearly a very nice feature of Rust to be able to rely on the fact that a
destructor for a type is always run (e.g. the `thread::scoped` API). Admitting
that destructors may not be run can lead to difficult API decisions later on and
even accidental unsafety. This route, however, is the least invasive for the
standard library and does not require radically changing types like `Rc` or
fast-tracking bug fixes to panicking destructors.

# Alternatives

The main alternative this proposal is to provide the guarantee that a destructor
for a type is always run and that it is memory unsafe to not do so. This would
require a number of pieces to work together:

* Panicking destructors not running other locals' destructors would [need to be
  fixed][dtor-bug1]
* Panics in the elements of containers would [need to be fixed][dtor-bug2] to
  continue running other elements' destructors.
* The `Rc` and `Arc` types would need be reevaluated somehow. One option would
  be to statically prevent cycles, and another option would be to disallow types
  that are unsafe to leak from being placed in `Rc` and `Arc` (more details
  below).
* An audit would need to be performed to ensure that there are no other known
  locations of leaks for types. There are likely more than one location than
  those listed here which would need to be addressed, and it's also likely that
  there would continue to be locations where destructors were not run.

There has been quite a bit of discussion specifically on the topic of `Rc` and
`Arc` as they may be tricky cases to fix. Specifically, the compiler could
perform some form of analysis could to forbid *all* cycles or just those that
would cause memory unsafety. Unfortunately, forbidding all cycles is likely to
be too limiting for `Rc` to be useful. Forbidding only "bad" cycles, however, is
a more plausible option.

Another alternative, as proposed by @arielb1, would be [a `Leak` marker
trait][leak] to indicate that a type is "safe to leak". Types like `Rc` would
require that their contents are `Leak`, and the `JoinGuard` type would opt-out
of it.  This marker trait could work similarly to `Send` where all types are
considered leakable by default, but types could opt-out of `Leak`. This
approach, however, requires `Rc` and `Arc` to have a `Leak` bound on their type
parameter which can often leak unfortunately into many generic contexts (e.g.
trait objects).  Another option would be to treak `Leak` more similarly to
`Sized` where all type parameters have a `Leak` bound by default. This change
may also cause confusion, however, by being unnecessarily restrictive (e.g. all
collections may want to take `T: ?Leak`).

[leak]: https://github.com/rust-lang/rust/issues/24292#issuecomment-91646130

Overall the changes necessary for this strategy are more invasive than admitting
destructors may not run, so this alternative is not proposed in this RFC.

# Unresolved questions

Are there remaining APIs in the standard library which rely on destructors being
run for memory safety?
