- Start Date: 2014-10-30
- RFC PR #: [rust-lang/rfcs#236](https://github.com/rust-lang/rfcs/pull/236)
- Rust Issue #: [rust-lang/rust#18466](https://github.com/rust-lang/rust/issues/18466)

# Summary

This is a *conventions* RFC for formalizing the basic conventions around error
handling in Rust libraries.

The high-level overview is:

* For *catastrophic errors*, abort the process or fail the task depending on
  whether any recovery is possible.

* For *contract violations*, fail the task. (Recover from programmer errors at a coarse grain.)

* For *obstructions to the operation*, use `Result` (or, less often,
  `Option`). (Recover from obstructions at a fine grain.)

* Prefer liberal function contracts, especially if reporting errors in input
  values may be useful to a function's caller.

This RFC follows up on [two](https://github.com/rust-lang/rfcs/pull/204)
[earlier](https://github.com/rust-lang/rfcs/pull/220) attempts by giving more
leeway in when to fail the task.

# Motivation

Rust provides two basic strategies for dealing with errors:

* *Task failure*, which unwinds to at least the task boundary, and by default
  propagates to other tasks through poisoned channels and mutexes. Task failure
  works well for coarse-grained error handling.

* *The Result type*, which allows functions to signal error conditions through
  the value that they return. Together with a lint and the `try!` macro,
  `Result` works well for fine-grained error handling.

However, while there have been some general trends in the usage of the two
handling mechanisms, we need to have formal guidelines in order to ensure
consistency as we stabilize library APIs. That is the purpose of this RFC.

For the most part, the RFC proposes guidelines that are already followed today,
but it tries to motivate and clarify them.

# Detailed design

Errors fall into one of three categories:

* Catastrophic errors, e.g. out-of-memory.
* Contract violations, e.g. wrong input encoding, index out of bounds.
* Obstructions, e.g. file not found, parse error.

The basic principle of the conventions is that:

* Catastrophic errors and programming errors (bugs) can and should only be
recovered at a *coarse grain*, i.e. a task boundary.
* Obstructions preventing an operation should be reported at a maximally *fine
grain* -- to the immediate invoker of the operation.

## Catastrophic errors

An error is _catastrophic_ if there is no meaningful way for the current task to
continue after the error occurs.

Catastrophic errors are _extremely_ rare, especially outside of `libstd`.

**Canonical examples**: out of memory, stack overflow.

### For catastrophic errors, fail the task.

For errors like stack overflow, Rust currently aborts the process, but
could in principle fail the task, which (in the best case) would allow
reporting and recovery from a supervisory task.

## Contract violations

An API may define a contract that goes beyond the type checking enforced by the
compiler. For example, slices support an indexing operation, with the contract
that the supplied index must be in bounds.

Contracts can be complex and involve more than a single function invocation. For
example, the `RefCell` type requires that `borrow_mut` not be called until all
existing borrows have been relinquished.

### For contract violations, fail the task.

A contract violation is always a bug, and for bugs we follow the Erlang
philosophy of "let it crash": we assume that software *will* have bugs, and we
design coarse-grained task boundaries to report, and perhaps recover, from these
bugs.

### Contract design

One subtle aspect of these guidelines is that the contract for a function is
chosen by an API designer -- and so the designer also determines what counts as
a violation.

This RFC does not attempt to give hard-and-fast rules for designing
contracts. However, here are some rough guidelines:

* Prefer expressing contracts through static types whenever possible.

* It *must* be possible to write code that uses the API without violating the
  contract.

* Contracts are most justified when violations are *inarguably* bugs -- but this
  is surprisingly rare.

* Consider whether the API client could benefit from the contract-checking
  logic.  The checks may be expensive. Or there may be useful programming
  patterns where the client does not want to check inputs before hand, but would
  rather attempt the operation and then find out whether the inputs were invalid.

* When a contract violation is the *only* kind of error a function may encounter
  -- i.e., there are no obstructions to its success other than "bad" inputs --
  using `Result` or `Option` instead is especially warranted. Clients can then use
  `unwrap` to assert that they have passed valid input, or re-use the error
  checking done by the API for their own purposes.

* When in doubt, use loose contracts and instead return a `Result` or `Option`.

## Obstructions

An operation is *obstructed* if it cannot be completed for some reason, even
though the operation's contract has been satisfied. Obstructed operations may
have (documented!) side effects -- they are not required to roll back after
encountering an obstruction.  However, they should leave the data structures in
a "coherent" state (satisfying their invariants, continuing to guarantee safety,
etc.).

Obstructions may involve external conditions (e.g., I/O), or they may involve
aspects of the input that are not covered by the contract.

**Canonical examples**: file not found, parse error.

### For obstructions, use `Result`

The
[`Result<T,E>` type](http://static.rust-lang.org/doc/master/std/result/index.html)
represents either a success (yielding `T`) or failure (yielding `E`). By
returning a `Result`, a function allows its clients to discover and react to
obstructions in a fine-grained way.

#### What about `Option`?

The `Option` type should not be used for "obstructed" operations; it
should only be used when a `None` return value could be considered a
"successful" execution of the operation.

This is of course a somewhat subjective question, but a good litmus
test is: would a reasonable client ever ignore the result? The
`Result` type provides a lint that ensures the result is actually
inspected, while `Option` does not, and this difference of behavior
can help when deciding between the two types.

Another litmus test: can the operation be understood as asking a
question (possibly with sideeffects)? Operations like `pop` on a
vector can be viewed as asking for the contents of the first element,
with the side effect of removing it if it exists -- with an `Option`
return value.

## Do not provide both `Result` and `fail!` variants.

An API should not provide both `Result`-producing and `fail`ing versions of an
operation. It should provide just the `Result` version, allowing clients to use
`try!` or `unwrap` instead as needed. This is part of the general pattern of
cutting down on redundant variants by instead using method chaining.

There is one exception to this rule, however. Some APIs are strongly oriented
around failure, in the sense that their functions/methods are explicitly
intended as assertions.  If there is no other way to check in advance for the
validity of invoking an operation `foo`, however, the API may provide a
`foo_catch` variant that returns a `Result`.

The main examples in `libstd` that *currently* provide both variants are:

* Channels, which are the primary point of failure propagation between tasks. As
  such, calling `recv()` is an _assertion_ that the other end of the channel is
  still alive, which will propagate failures from the other end of the
  channel. On the other hand, since there is no separate way to atomically test
  whether the other end has hung up, channels provide a `recv_opt` variant that
  produces a `Result`.

  > Note: the `_opt` suffix would be replaced by a `_catch` suffix if this RFC
  > is accepted.

* `RefCell`, which provides a dynamic version of the borrowing rules. Calling
  the `borrow()` method is intended as an assertion that the cell is in a
  borrowable state, and will `fail!` otherwise. On the other hand, there is no
  separate way to check the state of the `RefCell`, so the module provides a
  `try_borrow` variant that produces a `Result`.

  > Note: the `try_` prefix would be replaced by a `_catch` catch if this RFC is
  > accepted.

(Note: it is unclear whether these APIs will continue to provide both variants.)

# Drawbacks

The main drawbacks of this proposal are:

* Task failure remains somewhat of a landmine: one must be sure to document, and
  be aware of, all relevant function contracts in order to avoid task failure.

* The choice of what to make part of a function's contract remains somewhat
  subjective, so these guidelines cannot be used to decisively resolve
  disagreements about an API's design.

The alternatives mentioned below do not suffer from these problems, but have
drawbacks of their own.

# Alternatives

[Two](https://github.com/rust-lang/rfcs/pull/204)
[alternative](https://github.com/rust-lang/rfcs/pull/220) designs have been
given in earlier RFCs, both of which take a much harder line on using `fail!`
(or, put differently, do not allow most functions to have contracts).

As was
[pointed out by @SiegeLord](https://github.com/rust-lang/rfcs/pull/220#issuecomment-54715268),
however, mixing what might be seen as contract violations with obstructions can
make it much more difficult to write obstruction-robust code; see the linked
comment for more detail.

## Naming

There are numerous possible suffixes for a `Result`-producing variant:

* `_catch`, as proposed above. As
  [@kballard points out](https://github.com/rust-lang/rfcs/pull/236#issuecomment-55344336),
  this name connotes exception handling, which could be considered
  misleading. However, since it effectively prevents further unwinding, catching
  an exception may indeed be the right analogy.

* `_result`, which is straightforward but not as informative/suggestive as some
  of the other proposed variants.

* `try_` prefix. Also connotes exception handling, but has an unfortunately
  overlap with the common use of `try_` for nonblocking variants (which is in
  play for `recv` in particular).
