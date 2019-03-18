# `on_unimplemented`

The tracking issue for this feature is: [#29628]

[#29628]: https://github.com/rust-lang/rust/issues/29628

------------------------

The `on_unimplemented` feature provides the `#[rustc_on_unimplemented]`
attribute, which allows trait definitions to add specialized notes to error
messages when an implementation was expected but not found. You can refer
to the trait's generic arguments by name and to the resolved type using
`Self`.

For example:

```rust,compile_fail
#![feature(on_unimplemented)]

#[rustc_on_unimplemented="an iterator over elements of type `{A}` \
    cannot be built from a collection of type `{Self}`"]
trait MyIterator<A> {
    fn next(&mut self) -> A;
}

fn iterate_chars<I: MyIterator<char>>(i: I) {
    // ...
}

fn main() {
    iterate_chars(&[1, 2, 3][..]);
}
```

When the user compiles this, they will see the following;

```txt
error[E0277]: the trait bound `&[{integer}]: MyIterator<char>` is not satisfied
  --> <anon>:14:5
   |
14 |     iterate_chars(&[1, 2, 3][..]);
   |     ^^^^^^^^^^^^^ an iterator over elements of type `char` cannot be built from a collection of type `&[{integer}]`
   |
   = help: the trait `MyIterator<char>` is not implemented for `&[{integer}]`
   = note: required by `iterate_chars`
```

`on_unimplemented` also supports advanced filtering for better targeting
of messages, as well as modifying specific parts of the error message. You
target the text of:

 - the main error message (`message`)
 - the label (`label`)
 - an extra note (`note`)

For example, the following attribute

```rust,compile_fail
#[rustc_on_unimplemented(
    message="message",
    label="label",
    note="note"
)]
trait MyIterator<A> {
    fn next(&mut self) -> A;
}
```

Would generate the following output:

```text
error[E0277]: message
  --> <anon>:14:5
   |
14 |     iterate_chars(&[1, 2, 3][..]);
   |     ^^^^^^^^^^^^^ label
   |
   = note: note
   = help: the trait `MyIterator<char>` is not implemented for `&[{integer}]`
   = note: required by `iterate_chars`
```

To allow more targeted error messages, it is possible to filter the
application of these fields based on a variety of attributes when using
`on`:

 - `crate_local`: whether the code causing the trait bound to not be
   fulfilled is part of the user's crate. This is used to avoid suggesting
   code changes that would require modifying a dependency.
 - Any of the generic arguments that can be substituted in the text can be
   referred by name as well for filtering, like `Rhs="i32"`, except for
   `Self`.
 - `_Self`: to filter only on a particular calculated trait resolution, like
   `Self="std::iter::Iterator<char>"`. This is needed because `Self` is a
   keyword which cannot appear in attributes.
 - `direct`: user-specified rather than derived obligation.
 - `from_method`: usable both as boolean (whether the flag is present, like
   `crate_local`) or matching against a particular method. Currently used
   for `try`.
 - `from_desugaring`: usable both as boolean (whether the flag is present)
   or matching against a particular desugaring.

For example, the `Iterator` trait can be annotated in the following way:

```rust,compile_fail
#[rustc_on_unimplemented(
    on(
        _Self="&str",
        note="call `.chars()` or `.as_bytes()` on `{Self}"
    ),
    message="`{Self}` is not an iterator",
    label="`{Self}` is not an iterator",
    note="maybe try calling `.iter()` or a similar method"
)]
pub trait Iterator {}
```

Which would produce the following outputs:

```text
error[E0277]: `Foo` is not an iterator
 --> src/main.rs:4:16
  |
4 |     for foo in Foo {}
  |                ^^^ `Foo` is not an iterator
  |
  = note: maybe try calling `.iter()` or a similar method
  = help: the trait `std::iter::Iterator` is not implemented for `Foo`
  = note: required by `std::iter::IntoIterator::into_iter`

error[E0277]: `&str` is not an iterator
 --> src/main.rs:5:16
  |
5 |     for foo in "" {}
  |                ^^ `&str` is not an iterator
  |
  = note: call `.chars()` or `.bytes() on `&str`
  = help: the trait `std::iter::Iterator` is not implemented for `&str`
  = note: required by `std::iter::IntoIterator::into_iter`
```

If you need to filter on multiple attributes, you can use `all`, `any` or
`not` in the following way:

```rust,compile_fail
#[rustc_on_unimplemented(
    on(
        all(_Self="&str", T="std::string::String"),
        note="you can coerce a `{T}` into a `{Self}` by writing `&*variable`"
    )
)]
pub trait From<T>: Sized { /* ... */ }
```
