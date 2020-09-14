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

This lint **warns by default** and is **nightly-only**. This lint detects when
an intra-doc link fails to get resolved. For example:

```rust
/// I want to link to [`Inexistent`] but it doesn't exist!
pub fn foo() {}
```

You'll get a warning saying:

```text
error: `[`Inexistent`]` cannot be resolved, ignoring it...
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

## missing_crate_level_docs

This lint is **allowed by default**. It detects if there is no documentation
at the crate root. For example:

```rust
#![warn(missing_crate_level_docs)]
```

This will generate the following warning:

```text
warning: no documentation found for this crate's top-level module
  |
  = help: The following guide may be of use:
          https://doc.rust-lang.org/nightly/rustdoc/how-to-write-documentation.html
```

This is currently "allow" by default, but it is intended to make this a
warning in the future. This is intended as a means to introduce new users on
*how* to document their crate by pointing them to some instructions on how to
get started, without providing overwhelming warnings like `missing_docs`
might.

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

## invalid_codeblock_attributes

This lint **warns by default**. It detects code block attributes in
documentation examples that have potentially mis-typed values. For example:

```rust
/// Example.
///
/// ```should-panic
/// assert_eq!(1, 2);
/// ```
pub fn foo() {}
```

Which will give:

```text
warning: unknown attribute `should-panic`. Did you mean `should_panic`?
 --> src/lib.rs:1:1
  |
1 | / /// Example.
2 | | ///
3 | | /// ```should-panic
4 | | /// assert_eq!(1, 2);
5 | | /// ```
  | |_______^
  |
  = note: `#[warn(invalid_codeblock_attributes)]` on by default
  = help: the code block will either not be tested if not marked as a rust one or won't fail if it doesn't panic when running
```

In the example above, the correct form is `should_panic`. This helps detect
typo mistakes for some common attributes.
