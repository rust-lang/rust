- Feature Name: N/A
- Start Date: 2016-12-25
- RFC PR: https://github.com/rust-lang/rfcs/pull/1828
- Rust Issue: https://github.com/rust-lang/rust/issues/39588

# Summary
[summary]: #summary

Create a "Rust Bookshelf" of learning resources for Rust.

* Pull the book out of tree into `rust-lang/book`, which holds the second
  edition, currently.
* Pull the nomicon and the reference out of tree and convert them to mdBook.
* Pull the cargo docs out of tree and convert them to mdBook.
* Create a new "Nightly Book" in-tree.
* Provide a path forward for more long-form documentation to be maintained by
  the project.

This is largely about how doc.rust-lang.org is organized; today, it points to
the book, the reference, the nomicon, the error index, and the standard library
docs. This suggests unifying the first three into one thing.

# Motivation
[motivation]: #motivation

There are a few independent motivations for this RFC.

* Separate repos for separate projects.
* Consistency between long-form docs.
* A clear place for unstable documentation, which is now needed for
  stabilization.
* Better promoting good resources like the 'nomicon, which may not be as well
  known as "the book" is.

These will be discussed further in the detailed design.

# Detailed design
[design]: #detailed-design

Several new repositories will be made, one for each of:

* The Rustinomicon ("the 'nomicon")
* The Cargo Book
* The Rust Reference Manual

These would live under the `rust-lang` organization.

They will all use mdBook to build. They will have their existing text re-worked
into the format; at first a simple conversion, then more major improvements.
Their current text will be removed from the main tree.

The first edition of the book lives in-tree, but the second edition lives in
`rust-lang/book`. We'll remove the existing text from the tree and move it
into `rust-lang/book`.

A new book will be created from the "Nightly Rust" section of the book. It will
be called "The Nightly Book," and will contain unstable documentation for both
rustc and Cargo, as well as material that will end up in the reference. This
came up when [trying to document RFC
1623](https://github.com/rust-lang/rust/pull/37928). We don't have a unified
way of handling unstable documentation. This will give it a place to develop,
and part of the stabilization process will be moving documentation from this
book into the other parts of the documentation.

The nightly book will be organized around `#![feature]`s, so that you can look
up the documentation for each feature, as well as seeing which features
currently exist.

The nightly book is in-tree so that it runs more often, as part of people's
normal test suite. This doesn't mean that the book won't run on every commit;
just that the out-of-tree books will run mostly in CI, whereas the nightly
book will run when developers do `x.py check`. This is similar to how, today,
Traivs runs a subset of the tests, but buildbot runs all of them.

The landing page on doc.rust-lang.org will show off the full bookshelf, to let
people find the documenation they need. It will also link to their respective
repositories.

Finally, this creates a path for more books in the future: "the FFI Book" would
be one example of a possibility for this kind of thing. The docs team will
develop critera for accepting a book as part of the official project.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

The landing page on doc.rust-lang.org will show off the full bookshelf, to let
people find the documenation they need. It will also link to their respective
repositories.

# Drawbacks
[drawbacks]: #drawbacks

A ton of smaller repos can make it harder to find what goes where.

Removing work from `rust-lang/rust` means people aren't credited in release
notes any more. I will be opening a separate RFC to address this issue, it's
also an issue without this RFC being accepted.

Operations are harder, but they have to change to support this use-case for
other reasons, so this does not add any extra burden.

# Alternatives
[alternatives]: #alternatives

Do nothing.

Do only one part of this, instead of the whole thing.

Move all of the "bookshelf" into one repository, rather than individual ones.
This would require a lot more label-wrangling, but might be easier.

# Unresolved questions
[unresolved]: #unresolved-questions

How should the first and second editions of the book live in the same
repository?

What criteria should we use to accept new books?

Should we adopt "learning Rust with too many Linked Lists"?
