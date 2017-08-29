# Notes on work done during GSoC 2017 (by twk/ibabushkin)
The toplevel [README](https://github.com/ibabushkin/rust-semverver/blob/master/README.md)
outlines the functionality and usage of the project. This document complements it by
gathering references to the work that has been done during the Google Summer of Code 2017
and which eventually led to the current working state of the project.

## Completion status
All core functionality has been implemented. There are still some bugs present, whose
fixes depend on changes to [`rustc`](https://github.com/rust-lang/rust) that are currently
underway and possibly not yet merged at the time of submission.

See [this issue](https://github.com/ibabushkin/rust-semverver/issues/24) for a very rough
description of the problem and [this rust PR](https://github.com/rust-lang/rust/pull/43847)
as a reference to the fixes needed.

However, with the language being under active development and other changes taking place,
the project will need some future work: More bugfixes, and handling of new language
features might require a similar kind of development work as the one that took place over
the course of the program. Other possible enhancements are listed in the issue tracker:

* Checks for not directly code-related changes to a crate:
  [#8](https://github.com/ibabushkin/rust-semverver/issues/8)
* Recursive checking of dependencies:
  [#12](https://github.com/ibabushkin/rust-semverver/issues/12)
* Blacklisting modules and/or silencing analysis:
  [#22](https://github.com/ibabushkin/rust-semverver/issues/22)
* An automated tool checking all crates on `crates.io`:
  [#27](https://github.com/ibabushkin/rust-semverver/issues/27)

On a different front, the cargo plugin could need some technical improvements to improve
usability and code quality.

An overview of the functionality, and it's implementation can be found
[here](https://github.com/ibabushkin/rust-semverver/blob/master/doc/impl_notes.md).

## Progress made
The project provided a very through and challenging walkthrough to the internal working of
`rustc` and it's surrounding infrastructure. I had the opportunity to learn to approach
problems differently and in the context of a wider-reaching, larger project, which has
it's own priorities and forces a different approach. In that context, the provided
functionality is a stepping stone to maintain a codebase in and interact with the wider
rust internals community.

## List of references
* [this repository](https://github.com/ibabushkin/rust-semverver) contains the main body
  of code written.
* multiple pull requests to the main rust repository:
  * [#42507](https://github.com/rust-lang/rust/pull/42507) -- Fixes span translation in
    metadata decoding. Had to be amended by later changes to incorporate spans in error
    messages properly.
  * [#42593](https://github.com/rust-lang/rust/pull/42593) -- Implements the encoding of a
    reference to the source code in crate metadata, together with a lazy loading scheme.
    This provides for the source code to be rendered in error messages.
  * [#43128](https://github.com/rust-lang/rust/pull/43128) -- Allows to fold over type
    errors - which is a facility we then use.
  * [#43598](https://github.com/rust-lang/rust/pull/43598) -- A trivial oneliner to make
    an internal datatype more versatile for our purposes.
  * [#43739](https://github.com/rust-lang/rust/pull/43739) -- A fix to encode a more
    suited span in crate metadata for module items.
  * [#43847](https://github.com/rust-lang/rust/pull/43847) -- Not yet merged at the time
  of writing. Intends to allow for encoding of macro expansion information in crate
  metadata.
