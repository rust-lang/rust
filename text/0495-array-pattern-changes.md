- Start Date: 2014-12-03
- RFC PR: [rust-lang/rfcs#495](https://github.com/rust-lang/rfcs/pull/495)
- Rust Issue: [rust-lang/rust#23121](https://github.com/rust-lang/rust/issues/23121)

Summary
=======

Change array/slice patterns in the following ways:

- Make them only match on arrays (`[T; n]` and `[T]`), not slices;
- Make subslice matching yield a value of type `[T; n]` or `[T]`, not `&[T]` or
  `&mut [T]`;
- Allow multiple mutable references to be made to different parts of the same
  array or slice in array patterns (resolving rust-lang/rust [issue
  #8636](https://github.com/rust-lang/rust/issues/8636)).

Motivation
==========

Before DST (and after the removal of `~[T]`), there were only two types based on
`[T]`: `&[T]` and `&mut [T]`. With DST, we can have many more types based on
`[T]`, `Box<[T]>` in particular, but theoretically any pointer type around a
`[T]` could be used. However, array patterns still match on `&[T]`, `&mut [T]`,
and `[T; n]` only, meaning that to match on a `Box<[T]>`, one must first convert
it to a slice, which disallows moves. This may prove to significantly limit the
amount of useful code that can be written using array patterns.

Another problem with today’s array patterns is in subslice matching, which
specifies that the rest of a slice not matched on already in the pattern should
be put into a variable:

```rust
let foo = [1i, 2, 3];
match foo {
    [head, tail..] => {
        assert_eq!(head, 1);
        assert_eq!(tail, &[2, 3]);
    },
    _ => {},
}
```

This makes sense, but still has a few problems. In particular, `tail` is a
`&[int]`, even though the compiler can always assert that it will have a length
of `2`, so there is no way to treat it like a fixed-length array. Also, all
other bindings in array patterns are by-value, whereas bindings using subslice
matching are by-reference (even though they don’t use `ref`). This can create
confusing errors because of the fact that the `..` syntax is the only way of
taking a reference to something within a pattern without using the `ref`
keyword.

Finally, the compiler currently complains when one tries to take multiple
mutable references to different values within the same array in a slice pattern:

```rust
let foo: &mut [int] = &mut [1, 2, 3];
match foo {
    [ref mut a, ref mut b] => ...,
    ...
}
```

This fails to compile, because the compiler thinks that this would allow
multiple mutable borrows to the same value (which is not the case).

Detailed design
===============

- Make array patterns match only on arrays (`[T; n]` and `[T]`). For example,
  the following code:

  ```rust
  let foo: &[u8] = &[1, 2, 3];
  match foo {
      [a, b, c] => ...,
      ...
  }
  ```

  Would have to be changed to this:

  ```rust
  let foo: &[u8] = &[1, 2, 3];
  match foo {
      &[a, b, c] => ...,
      ...
  }
  ```

  This change makes slice patterns mirror slice expressions much more closely.

- Make subslice matching in array patterns yield a value of type `[T; n]` (if
  the array is of fixed size) or `[T]` (if not). This means changing most code
  that looks like this:

  ```rust
  let foo: &[u8] = &[1, 2, 3];
  match foo {
      [a, b, c..] => ...,
      ...
  }
  ```

  To this:

  ```rust
  let foo: &[u8] = &[1, 2, 3];
  match foo {
      &[a, b, ref c..] => ...,
      ...
  }
  ```

  It should be noted that if a fixed-size array is matched on using subslice
  matching, and `ref` is used, the type of the binding will be `&[T; n]`, *not*
  `&[T]`.

- Improve the compiler’s analysis of multiple mutable references to the same
  value within array patterns. This would be done by allowing multiple mutable
  references to different elements of the same array (including bindings from
  subslice matching):

  ```rust
  let foo: &mut [u8] = &mut [1, 2, 3, 4];
  match foo {
      &[ref mut a, ref mut b, ref c, ref mut d..] => ...,
      ...
  }
  ```

Drawbacks
=========

- This will break a non-negligible amount of code, requiring people to add `&`s
  and `ref`s to their code.

- The modifications to subslice matching will require `ref` or `ref mut` to be
  used in almost all cases. This could be seen as unnecessary.

Alternatives
============

- Do a subset of this proposal; for example, the modifications to subslice
  matching in patterns could be removed.

Unresolved questions
====================

- What are the precise implications to the borrow checker of the change to
  multiple mutable borrows in the same array pattern? Since it is a
  backwards-compatible change, it can be implemented after 1.0 if it turns out
  to be difficult to implement.
