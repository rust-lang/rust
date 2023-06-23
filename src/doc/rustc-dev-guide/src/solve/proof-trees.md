# Proof trees

The trait solver can optionally emit a "proof tree", a tree representation of what
happened while trying to prove a goal.

The used datastructures for which are currently stored in
[`rustc_middle::traits::solve::inspect`].

## What are they used for

There are 3 intended uses for proof trees. These uses are not yet implemented as
the representation of proof trees itself is currently still unstable.

They should be used by type system diagnostics to get information about
why a goal failed or remained ambiguous. They should be used by rustdoc to get the
auto-trait implementations for user-defined types, and they should be usable to
vastly improve the debugging experience of the trait solver.

For debugging you can use `-Zdump-solver-proof-tree` which dumps the proof tree
for all goals proven by the trait solver in the current session.

## Requirements and design constraints for proof trees

The trait solver uses [Canonicalization] and uses completely separate `InferCtxt` for
each nested goal. Both diagnostics and auto-traits in rustdoc need to correctly
handle "looking into nested goals". Given a goal like `Vec<Vec<?x>>: Debug`, we
canonicalize to `exists<T0> Vec<Vec<T0>>: Debug`, instantiate that goal as
`Vec<Vec<?0>>: Debug`, get a nested goal `Vec<?0>: Debug`, canonicalize this to get
`exists<T0> Vec<T0>: Debug`, instantiate this as `Vec<?0>: Debug` which then results
in a nested `?0: Debug` goal which is ambiguous.

We need to be able to figure out that `?x` corresponds to `?0` in the nested queries.

The debug output should also accurately represent the state at each point in the solver.
This means that even though a goal like `fn(?0): FnOnce(i32)` infers `?0` to `i32`, the
proof tree should still store `fn(<some infer var>): FnOnce(i32)` instead of
`fn(i32): FnOnce(i32)` until we actually infer `?0` to `i32`.

## The current implementation and how to extract information from proof trees.

Proof trees will be quite involved as they should accurately represent everything the
trait solver does, which includes fixpoint iterations and performance optimizations.

We intend to provide a lossy user interface for all usecases.

TODO: implement this user interface and explain how it can be used here.


[`rustc_middle::traits::solve::inspect`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/traits/solve/inspect/index.html
[Canonicalization]: ./canonicalization.md