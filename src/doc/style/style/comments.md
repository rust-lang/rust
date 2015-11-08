% Comments [FIXME: needs RFC]

### Avoid block comments.

Use line comments:

``` rust
// Wait for the main thread to return, and set the process error code
// appropriately.
```

Instead of:

``` rust
/*
 * Wait for the main thread to return, and set the process error code
 * appropriately.
 */
```

## Doc comments

Doc comments are prefixed by three slashes (`///`) and indicate
documentation that you would like to be included in Rustdoc's output.
They support
[Markdown syntax](https://en.wikipedia.org/wiki/Markdown)
and are the main way of documenting your public APIs.

The supported markdown syntax includes all of the extensions listed in the
[GitHub Flavored Markdown]
(https://help.github.com/articles/github-flavored-markdown) documentation,
plus superscripts.

### Summary line

The first line in any doc comment should be a single-line short sentence
providing a summary of the code. This line is used as a short summary
description throughout Rustdoc's output, so it's a good idea to keep it
short.

### Sentence structure

All doc comments, including the summary line, should begin with a
capital letter and end with a period, question mark, or exclamation
point. Prefer full sentences to fragments.

The summary line should be written in
[third person singular present indicative form]
(http://en.wikipedia.org/wiki/English_verbs#Third_person_singular_present).
Basically, this means write "Returns" instead of "Return".

For example:

``` rust
/// Sets up a default runtime configuration, given compiler-supplied arguments.
///
/// This function will block until the entire pool of M:N schedulers has
/// exited. This function also requires a local thread to be available.
///
/// # Arguments
///
/// * `argc` & `argv` - The argument vector. On Unix this information is used
///                     by `os::args`.
/// * `main` - The initial procedure to run inside of the M:N scheduling pool.
///            Once this procedure exits, the scheduling pool will begin to shut
///            down. The entire pool (and this function) will only return once
///            all child threads have finished executing.
///
/// # Return value
///
/// The return value is used as the process return code. 0 on success, 101 on
/// error.
```

### Code snippets

> **[FIXME]**

### Avoid inner doc comments.

Use inner doc comments _only_ to document crates and file-level modules:

``` rust
//! The core library.
//!
//! The core library is a something something...
```

### Explain context.

Rust doesn't have special constructors, only functions that return new
instances.  These aren't visible in the automatically generated documentation
for a type, so you should specifically link to them:

``` rust
/// An iterator that yields `None` forever after the underlying iterator
/// yields `None` once.
///
/// These can be created through
/// [`iter.fuse()`](trait.Iterator.html#method.fuse).
pub struct Fuse<I> {
    // ...
}
```
