# Lints

`rustdoc` provides lints to help you writing and testing your documentation. You
can use them like any other lints by doing this:

```rust,ignore
#![allow(missing_docs)] // allows the lint, no diagnostics will be reported
#![warn(missing_docs)] // warn if there are missing docs
#![deny(missing_docs)] // error if there are missing docs
```

Here is the list of the lints provided by `rustdoc`:

## broken_intra_doc_links

This lint **warns by default**. This lint detects when an [intra-doc link] fails to be resolved. For example:

[intra-doc link]: linking-to-items-by-name.md

```rust
/// I want to link to [`Nonexistent`] but it doesn't exist!
pub fn foo() {}
```

You'll get a warning saying:

```text
warning: unresolved link to `Nonexistent`
 --> test.rs:1:24
  |
1 | /// I want to link to [`Nonexistent`] but it doesn't exist!
  |                        ^^^^^^^^^^^^^ no item named `Nonexistent` in `test`
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
  = note: `#[warn(broken_intra_doc_links)]` on by default
help: to link to the enum, prefix with the item type
  |
1 | /// [`enum@Foo`]
  |      ^^^^^^^^^^
help: to link to the function, add parentheses
  |
1 | /// [`Foo()`]
  |      ^^^^^^^

```

## private_intra_doc_links

This lint **warns by default**. This lint detects when [intra-doc links] from public to private items.
For example:

```rust
/// [private]
pub fn public() {}
fn private() {}
```

This gives a warning that the link will be broken when it appears in your documentation:

```text
warning: public documentation for `public` links to private item `private`
 --> priv.rs:1:6
  |
1 | /// [private]
  |      ^^^^^^^ this item is private
  |
  = note: `#[warn(private_intra_doc_links)]` on by default
  = note: this link will resolve properly if you pass `--document-private-items`
```

Note that this has different behavior depending on whether you pass `--document-private-items` or not!
If you document private items, then it will still generate a link, despite the warning:

```text
warning: public documentation for `public` links to private item `private`
 --> priv.rs:1:6
  |
1 | /// [private]
  |      ^^^^^^^ this item is private
  |
  = note: `#[warn(private_intra_doc_links)]` on by default
  = note: this link resolves only because you passed `--document-private-items`, but will break without
```

[intra-doc links]: linking-to-items-by-name.html

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

## invalid_html_tags

This lint is **allowed by default** and is **nightly-only**. It detects unclosed
or invalid HTML tags. For example:

```rust
#![warn(invalid_html_tags)]

/// <h1>
/// </script>
pub fn foo() {}
```

Which will give:

```text
warning: unopened HTML tag `script`
 --> foo.rs:1:1
  |
1 | / /// <h1>
2 | | /// </script>
  | |_____________^
  |
  = note: `#[warn(invalid_html_tags)]` on by default

warning: unclosed HTML tag `h1`
 --> foo.rs:1:1
  |
1 | / /// <h1>
2 | | /// </script>
  | |_____________^

warning: 2 warnings emitted
```

## non_autolinks

This lint is **nightly-only** and **warns by default**. It detects links which
could use the "automatic" link syntax. For example:

```rust
/// http://example.org
/// [http://example.com](http://example.com)
/// [http://example.net]
///
/// [http://example.com]: http://example.com
pub fn foo() {}
```

Which will give:

```text
warning: this URL is not a hyperlink
 --> foo.rs:1:5
  |
1 | /// http://example.org
  |     ^^^^^^^^^^^^^^^^^^ help: use an automatic link instead: `<http://example.org>`
  |
  = note: `#[warn(non_autolinks)]` on by default

warning: unneeded long form for URL
 --> foo.rs:2:5
  |
2 | /// [http://example.com](http://example.com)
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: use an automatic link instead: `<http://example.com>`

warning: this URL is not a hyperlink
 --> foo.rs:3:6
  |
3 | /// [http://example.net]
  |      ^^^^^^^^^^^^^^^^^^ help: use an automatic link instead: `<http://example.net>`
```
