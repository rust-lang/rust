# Linking to items by name

Rustdoc is capable of directly linking to other rustdoc pages in Markdown documentation using the path of item as a link.

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

pub struct Bar;
```

You can refer to anything in scope, and use paths, including `Self`, `self`, `super`, and `crate`. You may also use `foo()` and `foo!()` to refer to methods/functions and macros respectively. Backticks around the link will be stripped.

```rust,edition2018
use std::sync::mpsc::Receiver;

/// This is an version of [`Receiver`], with support for [`std::future`].
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
/// This is a special implementation of [positional parameters]
///
/// [positional parameters]: std::fmt#formatting-parameters
struct MySpecialFormatter;
```

Paths in Rust have three namespaces: type, value, and macro. Items from these namespaces are allowed to overlap. In case of ambiguity, rustdoc will warn about the ambiguity and ask you to disambiguate, which can be done by using a prefix like  `struct@`, `enum@`, `type@`, `trait@`, `union@`, `const@`, `static@`, `value@`, `function@`, `mod@`, `fn@`, `module@`, `method@`, `prim@`, `primitive@`, `macro@`, or `derive@`:

```rust
/// See also: [`Foo`](struct@Foo)
struct Bar;

/// This is different from [`Foo`](fn@Foo)
struct Foo {}

fn Foo() {}
```

Note: Because of how `macro_rules` macros are scoped in Rust, the intra-doc links of a `macro_rules` macro will be resolved relative to the crate root, as opposed to the module it is defined in.
