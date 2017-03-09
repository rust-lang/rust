# Comments

Now that we have some functions, it’s a good idea to learn about comments.
Comments are notes that you leave to other programmers to help explain things
about your code. The compiler mostly ignores them.

Rust has two kinds of comments that you should care about: *line comments*
and *doc comments*.

```rust
// Line comments are anything after ‘//’ and extend to the end of the line.

let x = 5; // This is also a line comment.

// If you have a long explanation for something, you can put line comments next
// to each other. Put a space between the // and your comment so that it’s
// more readable.
```

The other kind of comment is a doc comment. Doc comments use `///` instead of
`//`, and support Markdown notation inside:

```rust
/// Adds one to the number given.
///
/// # Examples
///
/// ```
/// let five = 5;
///
/// assert_eq!(6, add_one(5));
/// # fn add_one(x: i32) -> i32 {
/// #     x + 1
/// # }
/// ```
fn add_one(x: i32) -> i32 {
    x + 1
}
```

There is another style of doc comment, `//!`, to comment containing items (e.g.
crates, modules or functions), instead of the items following it. Commonly used
inside crates root (lib.rs) or modules root (mod.rs):

```
//! # The Rust Standard Library
//!
//! The Rust Standard Library provides the essential runtime
//! functionality for building portable Rust software.
```

When writing doc comments, providing some examples of usage is very, very
helpful. You’ll notice we’ve used a new macro here: `assert_eq!`. This compares
two values, and `panic!`s if they’re not equal to each other. It’s very helpful
in documentation. There’s another macro, `assert!`, which `panic!`s if the
value passed to it is `false`.

You can use the [`rustdoc`](documentation.html) tool to generate HTML documentation
from these doc comments, and also to run the code examples as tests!
