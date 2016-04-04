- Feature Name: More API Documentation Conventions
- Start Date: 2016-03-31
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

[RFC 505] introduced certain conventions around documenting Rust projects. This RFC supersedes
that one, though it has the same aims: to describe how the Rust project should be documented,
and provide guidance for other Rust projects as well.

This RFC will contain some similar text as RFC 505, so that we can have one RFC with the full
conventions.

[RFC 505]: https://github.com/rust-lang/rfcs/blob/master/text/0505-api-comment-conventions.md

# Motivation
[motivation]: #motivation

Documentation is an extremely important part of any project. It’s important
that we have consistency in our documentation.

For the most part, the RFC proposes guidelines that are already followed today,
but it tries to motivate and clarify them.

# Detailed design
[design]: #detailed-design

This RFC is large. Here’s a table of contents:

* [Content](#content)
  * [Summary sentence](#summary-sentence)
  * [English](#english)
* [Form](#form)
  * [Use line comments](#use-line-comments)
  * [Using Markdown](#using-markdown)
* [Example](#example)

## Content
[content]: #content

These conventions relate to the contents of the documentation, the words themselves.

### Summary sentence
[summary-sentence]: #summary-sentence

In API documentation, the first line should be a single-line short sentence
providing a summary of the code. This line is used as a summary description
throughout Rustdoc’s output, so it’s a good idea to keep it short.

The summary line should be written in third person singular present indicative
form. Basically, this means write “Returns” instead of “Return.”

### English
[english]: #english

This section applies to `rustc` and the standard library.

All documentation is standardized on American English, with regards to
spelling, grammar, and punctuation conventions. Language changes over time,
so this doesn’t mean that there is always a correct answer to every grammar
question, but there is often some kind of formal consensus.

One specific rule that comes up often: when quoting something for emphasis,
use a single quote, and put punctuation outside the quotes, ‘this’. When
quoting something at length, “use double quotes and put the punctuation
inside of the quote.” Most documentation will end up using single quotes,
so if you’re not sure, just stick with them.

## Form
[form]: #form

These conventions relate to the formatting of the documentation, how they
appear in source code.

### Use line comments
[use-line-comments]: #use-line-comments

Avoid block comments. Use line comments instead:

```rust
// Wait for the main task to return, and set the process error code
// appropriately.
```

Instead of:

```rust
/*
 * Wait for the main task to return, and set the process error code
 * appropriately.
 */
```

Only use inner doc comments `//!` to write crate and module-level documentation,
nothing else. When using `mod` blocks, prefer `///` outside of the block:

```rust
/// This module contains tests
mod test {
    // ...
}
```

over

```rust
mod test {
    //! This module contains tests

    // ...
}
```

### Using Markdown
[using-markdown]: #using-markdown

Within doc comments, use Markdown to format your documentation.

Use top level headings # to indicate sections within your comment. Common headings:

* Examples
* Panics
* Errors
* Safety
* Aborts
* Undefined Behavior

Even if you only include one example, use the plural form: ‘Examples’ rather
than ‘Example’. Future tooling is easier this way.

Use graves (`) to denote a code fragment within a sentence.

Use triple graves (```) to write longer examples, like this:

    This code does something cool.

    ```rust
    let x = foo();

    x.bar();
    ```

When appropriate, make use of Rustdoc’s modifiers. Annotate triple grave blocks with
the appropriate formatting directive.

    ```rust
    println!("Hello, world!");
    ```

    ```ruby
    puts "Hello"
    ```

In API documentation, feel free to rely on the default being ‘rust’:

    /// For example:
    ///
    /// ```
    /// let x = 5;
    /// ```

In long-form documentation, always be explicit:

    For example:

    ```rust
    let x = 5;
    ```

This will highlight syntax in places that do not default to ‘rust’, like GitHub.

Rustdoc is able to test all Rust examples embedded inside of documentation, so
it’s important to mark what is not Rust so your tests don’t fail.

References and citation should be linked ‘reference style.’ Prefer

```
[Rust website]

[Rust website]: http://www.rust-lang.org
```

to

```
[Rust website](http://www.rust-lang.org)
```

### Examples in API docs
[examples-in-api-docs]: #examples-in-api-docs

Everything should have examples. Here is an example of how to do examples:

```
/// # Examples
///
/// Basic usage:
///
/// ```
/// use op;
///
/// let s = "foo";
/// let answer = op::compare(s, "bar");
/// ```
///
/// Passing a closure to compare with, rather than a string:
///
/// ```
/// use op;
///
/// let s = "foo";
/// let answer = op::compare(s, |a| a.chars().is_whitespace().all());
/// ```
```

For particularly simple APIs, still say “Examples” and “Basic usage:” for
consistency’s sake.

### Referring to types
[referring-to-types]: #referring-to-types

When talking about a type, use its full name. In other words, if the type is generic,
say `Option<T>`, not `Option`. An exception to this is lengthy bounds. Write `Cow<'a, B>`
rather than `Cow<'a, B> where B: 'a + ToOwned + ?Sized`.

Another possibility is to write in lower case using a more generic term. In other words,
‘string’ can refer to a `String` or an `&str`, and ‘an option’ can be ‘an `Option<T>`’.

### Link all the things
[link-all-the-things]: #link-all-the-things

A major drawback of Markdown is that it cannot automatically link types in API documentation.
Do this yourself with the reference-style syntax, for ease of reading:

```
/// The [`String`] passed in lorum ipsum...
///
/// [`String`]: ../string/struct.String.html
```

## Example
[example]: #example

Below is a full crate, with documentation following these rules. I am loosely basing
this off of my [ref_slice] crate, because it’s small, but I’m not claiming the code
is good here. It’s about the docs, not the code.

[ref_slice]: https://crates.io/crates/ref_slice

In lib.rs:

```rust
//! Turning references into slices
//!
//! This crate contains several utility functions for taking various kinds
//! of references and producing slices out of them. In this case, only full
//! slices, not ranges for sub-slices.
//!
//! # Layout
//!
//! At the top level, we have functions for working with references, `&T`.
//! There are two submodules for dealing with other types: `option`, for
//! &[`Option<T>`], and `mut`, for `&mut T`.
//!
//! [`Option<T>`]: http://doc.rust-lang.org/std/option/enum.Option.html

pub mod option;

/// Converts a reference to `T` into a slice of length 1.
///
/// This will not copy the data, only create the new slice.
///
/// # Panics
///
/// In this case, the code won’t panic, but if it did, the circumstances
/// in which it would would be included here.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// extern crate ref_slice;
/// use ref_slice::ref_slice;
/// 
/// let x = &5;
///
/// let slice = ref_slice(x);
///
/// assert_eq!(&[5], slice);
/// ```
///
/// A more compelx example. In this case, it’s the same example, because this
/// is a pretty trivial function, but use your imagination.
///
/// ```
/// extern crate ref_slice;
/// use ref_slice::ref_slice;
/// 
/// let x = &5;
///
/// let slice = ref_slice(x);
///
/// assert_eq!(&[5], slice);
/// ```
pub fn ref_slice<T>(s: &T) -> &[T] {
    unimplemented!()
}

/// Functions that operate on mutable references.
///
/// This submodule mirrors the parent module, but instead of dealing with `&T`,
/// they’re for `&mut T`.
mod mut {
    /// Converts a reference to `&mut T` into a mutable slice of length 1.
    ///
    /// This will not copy the data, only create the new slice.
    ///
    /// # Safety
    ///
    /// In this case, the code doesn’t need to be marked as unsafe, but if it
    /// did, the invariants you’re expected to uphold would be documented here.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// extern crate ref_slice;
    /// use ref_slice::mut;
    /// 
    /// let x = &mut 5;
    ///
    /// let slice = mut::ref_slice(x);
    ///
    /// assert_eq!(&mut [5], slice);
    /// ```
    pub fn ref_slice<T>(s: &mut T) -> &mut [T] {
        unimplemented!()
    }
}
```

in `option.rs`:

```rust
//! Functions that operate on references to [`Option<T>`]s.
//!
//! This submodule mirrors the parent module, but instead of dealing with `&T`,
//! they’re for `&`[`Option<T>`].
//!
//! [`Option<T>`]: http://doc.rust-lang.org/std/option/enum.Option.html

/// Converts a reference to `Option<T>` into a slice of length 0 or 1.
///
/// [`Option<T>`]: http://doc.rust-lang.org/std/option/enum.Option.html
///
/// This will not copy the data, only create the new slice.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// extern crate ref_slice;
/// use ref_slice::option;
/// 
/// let x = &Some(5);
///
/// let slice = option::ref_slice(x);
///
/// assert_eq!(&[5], slice);
/// ```
///
/// `None` will result in an empty slice:
///
/// ```
/// extern crate ref_slice;
/// use ref_slice::option;
/// 
/// let x: &Option<i32> = &None;
///
/// let slice = option::ref_slice(x);
///
/// assert_eq!(&[], slice);
/// ```
pub fn ref_slice<T>(opt: &Option<T>) -> &[T] {
    unimplemented!()
}
```

## Formatting

# Drawbacks
[drawbacks]: #drawbacks

It’s possible that RFC 505 went far enough, and something this detailed is inappropriate.

# Alternatives
[alternatives]: #alternatives

We could stick with the more minimal conventions of the previous RFC.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
