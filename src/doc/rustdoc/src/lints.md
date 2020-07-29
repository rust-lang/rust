# Lints

`rustdoc` provides lints to help you writing and testing your documentation. You
can use them like any other lints by doing this:

```rust,ignore
#![allow(missing_docs)] // allowing the lint, no message
#![warn(missing_docs)] // warn if there is missing docs
#![deny(missing_docs)] // rustdoc will fail if there is missing docs
```

Here is the list of the lints provided by `rustdoc`:

## broken_intra_doc_links

This lint **warns by default**. This lint detects when an [intra-doc link] fails to get resolved. For example:

 [intra-doc link]: linking-to-items-by-name.html

```rust
/// I want to link to [`Nonexistent`] but it doesn't exist!
pub fn foo() {}
```

You'll get a warning saying:

```text
warning: `[Nonexistent]` cannot be resolved, ignoring it.
 --> test.rs:1:24
  |
1 | /// I want to link to [`Nonexistent`] but it doesn't exist!
  |                        ^^^^^^^^^^^^^ cannot be resolved, ignoring
```

It will also warn when there is an ambiguity and suggest how to disambiguate:

```rust
/// [`Foo`]
pub fn function() {}

pub enum Foo {}

pub fn Foo(){}
```

```text
warning: `Foo` is both an enum and a function
 --> test.rs:1:6
  |
1 | /// [`Foo`]
  |      ^^^^^ ambiguous link
  |
  = note: `#[warn(intra_doc_link_resolution_failure)]` on by default
help: to link to the enum, prefix with the item type
  |
1 | /// [`enum@Foo`]
  |      ^^^^^^^^^^
help: to link to the function, add parentheses
  |
1 | /// [`Foo()`]
  |      ^^^^^^^

```

## missing_docs

This lint is **allowed by default**. It detects items missing documentation.
For example:

```rust
#![warn(missing_docs)]

pub fn undocumented() {}
# fn main() {}
```

The `undocumented` function will then have the following warning:

```text
warning: missing documentation for a function
  --> your-crate/lib.rs:3:1
   |
 3 | pub fn undocumented() {}
   | ^^^^^^^^^^^^^^^^^^^^^
```

## missing_doc_code_examples

This lint is **allowed by default** and is **nightly-only**. It detects when a documentation block
is missing a code example. For example:

```rust
#![warn(missing_doc_code_examples)]

/// There is no code example!
pub fn no_code_example() {}
# fn main() {}
```

The `no_code_example` function will then have the following warning:

```text
warning: Missing code example in this documentation
  --> your-crate/lib.rs:3:1
   |
LL | /// There is no code example!
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

To fix the lint, you need to add a code example into the documentation block:

```rust
/// There is no code example!
///
/// ```
/// println!("calling no_code_example...");
/// no_code_example();
/// println!("we called no_code_example!");
/// ```
pub fn no_code_example() {}
```

## private_doc_tests

This lint is **allowed by default**. It detects documentation tests when they
are on a private item. For example:

```rust
#![warn(private_doc_tests)]

mod foo {
    /// private doc test
    ///
    /// ```
    /// assert!(false);
    /// ```
    fn bar() {}
}
# fn main() {}
```

Which will give:

```text
warning: Documentation test in private item
  --> your-crate/lib.rs:4:1
   |
 4 | /     /// private doc test
 5 | |     ///
 6 | |     /// ```
 7 | |     /// assert!(false);
 8 | |     /// ```
   | |___________^
```
