# Introduction

This document is the primary reference for the Rust programming language. It
provides three kinds of material:

  - Chapters that informally describe each language construct and their use.
  - Chapters that informally describe the memory model, concurrency model,
    runtime services, linkage model and debugging facilities.
  - Appendix chapters providing rationale and references to languages that
    influenced the design.

This document does not serve as an introduction to the language. Background
familiarity with the language is assumed. A separate [book] is available to
help acquire such background familiarity.

This document also does not serve as a reference to the [standard] library
included in the language distribution. Those libraries are documented
separately by extracting documentation attributes from their source code. Many
of the features that one might expect to be language features are library
features in Rust, so what you're looking for may be there, not here.

Finally, this document is not normative. It may include details that are
specific to `rustc` itself, and should not be taken as a specification for
the Rust language. We intend to produce such a document someday, but this
is what we have for now.

You may also be interested in the [grammar].

[book]: ../book/index.html
[standard]: ../std/index.html
[grammar]: ../grammar.html
