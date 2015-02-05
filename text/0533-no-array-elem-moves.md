- Start Date: 2014-12-19
- RFC PR: [rust-lang/rfcs#533](https://github.com/rust-lang/rfcs/pull/533)
- Rust Issue: [rust-lang/rust#21963](https://github.com/rust-lang/rust/issues/21963)

# Summary

In order to prepare for an expected future implementation of
[non-zeroing dynamic drop], remove support for:

* moving individual elements into an *uninitialized* fixed-sized array, and

* moving individual elements out of fixed-sized arrays `[T; n]`,
  (copying and borrowing such elements is still permitted).

[non-zeroing dynamic drop]: https://github.com/rust-lang/rfcs/pull/320

# Motivation

If we want to continue supporting dynamic drop while also removing
automatic memory zeroing and drop-flags, then we need to either (1.)
adopt potential complex code generation strategies to support arrays
with only *some* elements initialized (as discussed in the [unresolved
questions for RFC PR 320], or we need to (2.) remove support for
constructing such arrays in safe code.

[unresolved questions for RFC PR 320]: https://github.com/pnkfelix/rfcs/blob/6288739c584ee6830aa0f79f983c5e762269c562/active/0000-nonzeroing-dynamic-drop.md#how-to-handle-moves-out-of-arrayindex_expr

This RFC is proposing the second tack.

The expectation is that relatively few libraries need to support
moving out of fixed-sized arrays (and even fewer take advantage of
being able to initialize individual elements of an uninitialized
array, as supporting this was almost certainly not intentional in the
language design). Therefore removing the feature from the language
will present relatively little burden.

# Detailed design

If an expression `e` has type `[T; n]` and `T` does not implement
`Copy`, then it will be illegal to use `e[i]` in an r-value position.

If an expression `e` has type `[T; n]` expression `e[i] = <expr>`
will be made illegal at points in the control flow where `e` has not
yet been initialized.

Note that it *remains* legal to overwrite an element in an initialized
array: `e[i] = <expr>`, as today.  This will continue to drop the
overwritten element before moving the result of `<expr>` into place.

Note also that the proposed change has no effect on the semantics of
destructuring bind; i.e. `fn([a, b, c]: [Elem; 3]) { ... }` will
continue to work as much as it does today.

A prototype implementation has been posted at [Rust PR 21930].

[Rust PR 21930]: https://github.com/rust-lang/rust/pull/21930

# Drawbacks

* Adopting this RFC is introducing a limitation on the language based
  on a hypothetical optimization that has not yet been implemented
  (though much of the ground work for its supporting analyses are
  done).

Also, as noted in the [comment thread from RFC PR 320]

[comment thread from RFC PR 320]: https://github.com/rust-lang/rfcs/pull/320#issuecomment-59533551

* We support moving a single element out of an n-tuple, and "by
  analogy" we should support moving out of `[T; n]`
  (Note that one can still move out of `[T; n]` in some cases
  via destructuring bind.)

* It is "nice" to be able to write
  ```rust
  fn grab_random_from(actions: [Action; 5]) -> Action { actions[rand_index()] }
  ```
  to express this now, one would be forced to instead use clone() (or
  pass in a `Vec` and do some element swapping).


# Alternatives

We can just leave things as they are; there are hypothetical
code-generation strategies for supporting non-zeroing drop even with
this feature, as discussed in the [comment thread from RFC PR 320].

# Unresolved questions

None

