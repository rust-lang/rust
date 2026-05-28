# `doc_notable_trait`

The tracking issue for this feature is: [#45040]

The `doc_notable_trait` feature allows the use of the `#[doc(notable_trait)]`
attribute, which will display the trait in a "Notable traits" dialog for
functions returning types that implement the trait. For example, this attribute
is applied to the `Iterator`, `Future`, `io::Read`, and `io::Write` traits in
the standard library.

You can do this on your own traits like so:

```
#![feature(doc_notable_trait)]

#[doc(notable_trait)]
pub trait MyTrait {}

pub struct MyStruct;
impl MyTrait for MyStruct {}

/// The docs for this function will have a button that displays a dialog about
/// `MyStruct` implementing `MyTrait`.
pub fn my_fn() -> MyStruct { MyStruct }
```

This feature was originally implemented in PR [#45039].

See also its documentation in [the rustdoc book][rustdoc-book-notable_trait].

[#45040]: https://github.com/rust-lang/rust/issues/45040
[#45039]: https://github.com/rust-lang/rust/pull/45039
[rustdoc-book-notable_trait]: ../../rustdoc/unstable-features.html#adding-your-trait-to-the-notable-traits-dialog
