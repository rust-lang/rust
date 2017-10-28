# `on_unimplemented`

The tracking issue for this feature is: [#29628]

[#29628]: https://github.com/rust-lang/rust/issues/29628

------------------------

The `on_unimplemented` feature provides the `#[rustc_on_unimplemented]`
attribute, which allows trait definitions to add specialized notes to error
messages when an implementation was expected but not found.

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

error: aborting due to previous error
```

