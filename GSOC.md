# Notes on work done during GSoC 2017
The toplevel [README](https://github.com/ibabushkin/rust-semverver/blob/master/README.md)
outlines the functionality and usage of the project. This document complements it by
gathering references to the work that has been done during the Google Summer of Code 2017
and which eventually led to the current working state of the project.

## Completion status
All core functioanlity has been implemented. There are still some bugs present, whose
fixes depend on changes to [rustc](https://github.com/rust-lang/rust) that are currently
underway and possibly not yet merged at the time of submission.

See [this issue](https://github.com/ibabushkin/rust-semverver/issues/24) for a very rough
description of the problem and [this rust PR](https://github.com/rust-lang/rust/pull/43847)
as a reference to the fixes needed.

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
