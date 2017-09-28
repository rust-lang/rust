<!---
Copyright 2017 The Rust Project Developers. See the COPYRIGHT
file at the top-level directory of this distribution.

Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
<LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
option. This file may not be copied, modified, or distributed
except according to those terms.
-->

- Feature Name: external_doc
- Start Date: 2017-04-26
- RFC PR: https://github.com/rust-lang/rfcs/pull/1990
- Rust Issue: https://github.com/rust-lang/rust/issues/44732

# Summary
[summary]: #summary

Documentation is an important part of any project, it allows developers to
explain how to use items within a library as well as communicate the intent of
how to use it through examples. Rust has long championed this feature through
the use of documentation comments and `rustdoc` to generate beautiful, easy to
navigate documentation. However, there is no way right now to have documentation
be imported into the code from an external file. This RFC proposes a way to
extend the functionality of Rust to include this ability.

# Motivation
[motivation]: #motivation

1. Many smaller crates are able to do all of the documentation that's needed in
   a README file within their repo. Being able to include this as a crate or
   module level doc comment would mean not having to duplicate documentation and
   is easier to maintain. This means that one could run `cargo doc` with the
   small crate as a dependency and be able to access the contents of the README
   without needing to go online to the repo to read it. This also would help
   with [this issue on
   crates.io](https://github.com/rust-lang/crates.io/issues/81) by making it
   easy to have the README in the crate and the crate root at the same.
2. The feature would provide a way to have easier to read code for library
   maintainers. Sometimes doc comments are quite long in terms of line count
   (items in
   [libstd](https://github.com/rust-lang/rust/blob/master/src/libstd) are a good
   example of this). Doc comments document behavior of functions, structs, and
   types to the end user, they do not explain for a coder working on the library
   as to how they work internally. When actually writing code for a
   library the doc comments end up cluttering the source code making it harder
   to find relevant lines to change or skim through and read what is going on.
3. Localization is something else that would further open up access to the
   community. By providing docs in different languages we could significantly
   expand our reach as a community and be more inclusive of those where English
   is not their first language. This would be made possible with a config flag
   choosing what file to import as a doc comment.

These are just a few reasons as to why we should do this, but the expected
outcome of this feature is expected to be positive with little to no downside
for a user.

# Detailed Design
[design]: #detailed-design

All files included through the attribute will be relative paths from the crate
root directory. Given a file like this stored in `docs/example.md`:

```md
# I'm an example
This is a markdown file that gets imported to Rust as a Doc comment.
```
where `src` is in the same directory as `docs`. Given code like this:

```rust
#[doc(include = "../docs/example.md")]
fn my_func() {
  // Hidden implementation
}
```

It should expand to this at compile time:

```rust
#[doc("# I'm an example\nThis is a markdown file that gets imported to Rust as a doc comment.")]
fn my_func() {
  // Hidden implementation
}
```

Which `rustdoc` should be able to figure out and use for documentation.

If the code is written like this:

```rust
#![doc(include = "../docs/example.md")]
fn my_func() {
  // Hidden implementation
}
```

It should expand out to this at compile time:

```rust
#![doc("# I'm an example\nThis is a markdown file that gets imported to Rust as a doc comment.")]
fn my_func() {
  // Hidden implementation
}
```

In the case of this code:

```rust
mod example {
    #![doc(include = "../docs/example.md")]
    fn my_func() {
      // Hidden implementation
    }
}
```

It should expand out to:

```rust
mod example {
    #![doc("# I'm an example\nThis is a markdown file that gets imported to Rust as a doc comment.")]
    fn my_func() {
      // Hidden implementation
    }
}
```

## Acceptable Paths

If you've noticed the path given `../docs/example.md` is a relative path to
`src`. This was decided upon as a good first implementation and further RFCs
could be written to expand on what syntax is acceptable for paths. For instance
not being relative to `src`.

## Missing Files or Incorrect Paths
If a file given to `include` is missing then this should trigger a compilation
error as the given file was supposed to be put into the code but for some reason
or other it is not there.

## Line Numbers When Errors Occur
As with all macros being expanded this brings up the question of line numbers
and for documentation tests especially so, to keep things simple for the user
the documentation should be treated separately from the code. Since the
attribute only needs to be expanded with `rustdoc` or `cargo test`, it should be
ignored by the compiler except for having the proper lines for error messages.

For example if we have this:

```rust
#[doc(include = "../docs/example.md")] // Line 1
f my_func() {                          // Line 2
  // Hidden implementation             // Line 3
}                                      // Line 4
```

Then we would have a syntax error on line 2, however the doc comment comes
before that. In this case the compiler would ignore the attribute for expansion,
but would say that the error occurs on line 2 rather than saying it is line 1 if
the attribute is ignored. This makes it easy for the user to spot their error.
This same behavior should be observed in the case of inline tests and those in
the tests directory.

If we have a documentation test failure the line number should be for the
external doc file and the line number where it fails, rather than a line number
from the code base itself. Having the numbers for the lines being used because
they were inserted into the code for these scenarios would cause confusion and
would obfuscate where errors occur, making it harder not easier for end users,
making this feature useless if it creates ergonomic overhead like this.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

`#[doc(include = "file_path")]` is an extension of the current
`#[doc = "doc"]` attribute by allowing documentation to exist outside of the
source code. This isn't entirely hard to grasp if one is familiar with
attributes but if not then this syntax vs a `///` or `//!` type of comment
could cause confusion. By labeling the attribute as `external_doc`, having a
clear path and type (either `line` or `mod`) then should, at the very least,
provide context as to what's going on and where to find this file for inclusion.

The acceptance of this proposal would minimally impact all levels of Rust users
as it is something that provides convenience but is not a necessary thing to
learn to use Rust. It should be taught to existing users by updating
documentation to show it in use and to include in in the Rust Programming
Language book to teach new users. Currently the newest version of The Rust
Programming Language book has a section for [doc comments](https://doc.rust-lang.org/nightly/book/second-edition/ch14-02-publishing-to-crates-io.html#documentation-comments) that will need to be expanded
to show how users can include docs from external sources. The Rust Reference
comments section would need to updated to include this new syntax as well.

# Drawbacks
[drawbacks]: #drawbacks

- This might confuse or frustrate people reading the code directly who prefer
  those doc comments to be inline with the code rather than in a separate file.
  This creates a burden of ergonomics by having to know the context of the code
  that the doc comment is for while reading it separately from the code it
  documents.

# Alternatives
[alternatives]: #alternatives

Currently there already [exists a plugin](https://github.com/mgattozzi/rdoc)
that could be used as a reference and has shown that
[there is interest](https://www.reddit.com/r/rust/comments/67kqs6/announcing_rdoc_a_tiny_rustc_plugin_to_host_your/).
Some limitations though being that it did not have module doc support and it
would make doc test failures unclear as to where they happened, which could be
solved with better support and intrinsics from the compiler.

This same idea could be implemented as a crate with procedural macros (which are
on nightly now) so that others can opt in to this rather than have it be part of
the language itself. Docs will remain the same as they always have and will
continue to work as is if this alternative is chosen, though this means we limit
what we do and do not want rustc/rustdoc to be able to achieve here when it
comes to docs.

# Unresolved questions
[unresolved]: #unresolved-questions

- What would be best practices for adding docs to crates?
