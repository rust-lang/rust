# Linking to items by name

Rustdoc is capable of directly linking to other rustdoc pages using the path of
the item as a link.

For example, in the following code all of the links will link to the rustdoc page for `Bar`:

```rust
/// This struct is not [Bar]
pub struct Foo1;

/// This struct is also not [bar](Bar)
pub struct Foo2;

/// This struct is also not [bar][b]
///
/// [b]: Bar
pub struct Foo3;

/// This struct is also not [`Bar`]
pub struct Foo4;

/// This struct *is* [`Bar`]!
pub struct Bar;
```

Backticks around the link will be stripped, so ``[`Option`]`` will correctly
link to `Option`.

You can refer to anything in scope, and use paths, including `Self`, `self`,
`super`, and `crate`. You may also use `foo()` and `foo!()` to refer to methods/functions and macros, respectively.

You can also refer to items with generic parameters like `Vec<T>`. The link will
resolve as if you had written ``[`Vec<T>`](Vec)``. Fully-qualified syntax (for example,
`<Vec as IntoIterator>::into_iter()`) is [not yet supported][fqs-issue], however.

[fqs-issue]: https://github.com/rust-lang/rust/issues/74563

```rust,edition2018
use std::sync::mpsc::Receiver;

/// This is a version of [`Receiver<T>`] with support for [`std::future`].
///
/// You can obtain a [`std::future::Future`] by calling [`Self::recv()`].
pub struct AsyncReceiver<T> {
    sender: Receiver<T>
}

impl<T> AsyncReceiver<T> {
    pub async fn recv() -> T {
        unimplemented!()
    }
}
```

You can also link to sections using URL fragment specifiers:

```rust
/// This is a special implementation of [positional parameters].
///
/// [positional parameters]: std::fmt#formatting-parameters
struct MySpecialFormatter;
```

Paths in Rust have three namespaces: type, value, and macro. Item names must be
unique within their namespace, but can overlap with items outside of their
namespace. In case of ambiguity, rustdoc will warn about the ambiguity and ask you to disambiguate, which can be done by using a prefix like `struct@`, `enum@`, `type@`, `trait@`, `union@`, `const@`, `static@`, `value@`, `fn@`, `function@`, `mod@`, `module@`, `method@`, `prim@`, `primitive@`, `macro@`, or `derive@`:

```rust
/// See also: [`Foo`](struct@Foo)
struct Bar;

/// This is different from [`Foo`](fn@Foo)
struct Foo {}

fn Foo() {}
```

You can also disambiguate for functions by adding `()` after the function name,
or for macros by adding `!` after the macro name:

```rust
/// See also: [`Foo`](struct@Foo)
struct Bar;

/// This is different from [`Foo()`]
struct Foo {}

fn Foo() {}
```

Note: Because of how `macro_rules!` macros are scoped in Rust, the intra-doc links of a `macro_rules!` macro will be resolved [relative to the crate root][#72243], as opposed to the module it is defined in.

[#72243]: https://github.com/rust-lang/rust/issues/72243
