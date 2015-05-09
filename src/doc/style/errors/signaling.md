% Signaling errors [RFC #236]

> The guidelines below were approved by [RFC #236](https://github.com/rust-lang/rfcs/pull/236).

Errors fall into one of three categories:

* Catastrophic errors, e.g. out-of-memory.
* Contract violations, e.g. wrong input encoding, index out of bounds.
* Obstructions, e.g. file not found, parse error.

The basic principle of the convention is that:

* Catastrophic errors and programming errors (bugs) can and should only be
recovered at a *coarse grain*, i.e. a thread boundary.
* Obstructions preventing an operation should be reported at a maximally *fine
grain* -- to the immediate invoker of the operation.

## Catastrophic errors

An error is _catastrophic_ if there is no meaningful way for the current thread to
continue after the error occurs.

Catastrophic errors are _extremely_ rare, especially outside of `libstd`.

**Canonical examples**: out of memory, stack overflow.

### For catastrophic errors, panic

For errors like stack overflow, Rust currently aborts the process, but
could in principle panic, which (in the best case) would allow
reporting and recovery from a supervisory thread.

## Contract violations

An API may define a contract that goes beyond the type checking enforced by the
compiler. For example, slices support an indexing operation, with the contract
that the supplied index must be in bounds.

Contracts can be complex and involve more than a single function invocation. For
example, the `RefCell` type requires that `borrow_mut` not be called until all
existing borrows have been relinquished.

### For contract violations, panic

A contract violation is always a bug, and for bugs we follow the Erlang
philosophy of "let it crash": we assume that software *will* have bugs, and we
design coarse-grained thread boundaries to report, and perhaps recover, from these
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

## Do not provide both `Result` and `panic!` variants.

An API should not provide both `Result`-producing and `panic`king versions of an
operation. It should provide just the `Result` version, allowing clients to use
`try!` or `unwrap` instead as needed. This is part of the general pattern of
cutting down on redundant variants by instead using method chaining.
