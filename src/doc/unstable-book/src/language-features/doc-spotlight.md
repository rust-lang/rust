# `doc_spotlight`

The tracking issue for this feature is: [#45040]

The `doc_spotlight` feature allows the use of the `spotlight` parameter to the `#[doc]` attribute,
to "spotlight" a specific trait on the return values of functions. Adding a `#[doc(spotlight)]`
attribute to a trait definition will make rustdoc print extra information for functions which return
a type that implements that trait. This attribute is applied to the `Iterator`, `io::Read`, and
`io::Write` traits in the standard library.

You can do this on your own traits, like this:

```
#![feature(doc_spotlight)]

#[doc(spotlight)]
pub trait MyTrait {}

pub struct MyStruct;
impl MyTrait for MyStruct {}

/// The docs for this function will have an extra line about `MyStruct` implementing `MyTrait`,
/// without having to write that yourself!
pub fn my_fn() -> MyStruct { MyStruct }
```

This feature was originally implemented in PR [#45039].

[#45040]: https://github.com/rust-lang/rust/issues/45040
[#45039]: https://github.com/rust-lang/rust/pull/45039
