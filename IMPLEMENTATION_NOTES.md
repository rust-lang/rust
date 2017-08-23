# Implementation Notes
This document provides a high-level overview over the project's structure and
implementation, together with explanations of the various implementation decisions that
have been taken.

The toplevel directory of the repository is structured according to cargo's conventions:

* `src` contains the main project source code.
* `tests` contains integration tests (more on these in their dedicated section).
* only one crate, `semverver` is provided. It provides two binaries, whose functionality
  is elaborated later on. The actual functionality is currently not exposed as a library,
  but this change is trivial to implement.
* a cargo manifest and lockfile, various documentation material you are currently reading,
  etc. is also placed at the top level.

## Source code structure
Inside the `src` subdirectory, the main functionality can be found inside the `semcheck`
directory, while the `bin` directory contains the two executables provided by the crate.

### Execution overview
The provided binaries are a cargo plugin and a custom rustc driver, respectively, and
allow to analyze local and remote crate pairs for semver compatibility.

A typical invocation, assuming that both binaries are on the user's `PATH`, is performed
by invoking `cargo semver` in a source code repository that can be built with cargo. This
invokes cargo's plugin mechanism, that then passes a set of command line arguments to
`cargo-semver`. This is the binary responsible to determine and compile the analysis
targets, whose inner workings are currently quite simplistic and allow for any combination
of "local" and "remote" crates - that is, source code repositories available through the
filesystem and from `crates.io`, defaulting to the current directory and the last
published version on `crates.io`, respectively. When a fitting pair of crates has been
compiled, the compiler driver, located in the `rust-semverver` binary, is invoked on a
dummy crate linking both versions as `old` and `new`. All further analysis is performed in
the compiler driver.

To be more precise, the compiler driver runs the regular compilation machinery up to the
type checking phase and passes control to our analysis code, aborting the compilation
afterwards.

This overall design has been chosen because it allows us to work on the data structures
representing parts of both the old and the new crate from the same compiler instance,
which simplifies the process by a great margin. Also, type information on all items being
analyzed is vital and has to be without any contradiction - so basing the analysis on
successfully compiled code is natural. The drawback, however, is that the needed
information is only available in a slightly manipulated form, since it has been encoded in
library metadata and decoded afterwards. This required some changes to the compiler's
metadata handling that have been mostly upstreamed by now. Another weak point is the
performance penalty imposed by two compilations and an analysis run on the target crate,
but this is very hard to circumvent, as is the necessity of using a nightly rust compiler
to use the tool - much alike to `rust-clippy`.

### Analysis overview
The actual analysis is separated in multiple passes, whose main entry point is the
`run_analysis` function in the `traverse` submodule in `semverver::semcheck`. These passes
are structured as follows:

1. Named items are matched up and checked for structural changes in a module traversal
   scheme. Structural changes are changes to ADT structure, or additions and removals of
   items, type and region parameters, changes to item visibility and (re)export structure,
   and similar miscellaneous changes to the code being analyzed.
2. Not yet matched hidden items are opportunistically matched based on their usage in
   public items' types. This is implemented in the `mismatch` submodule.
3. All items which haven't undergone breaking changes are checked for changes to their
   trait bounds and (if applicable) types. This requires a translation of the analyzed old
   item into the new crate using the previously established correspondence between items.
   That mechanism is implemented in the `translate` submodule, and used very intensively
   throughout the last two passes. Translation is based on item correspondence, which is
   kept track of in the `mapping` submodule.
4. Inherent items and trait impls are matched up, if possible. This, too requires the
   translation of bounds and types of the old item. However, to determine non-breaking
   changes, bounds checks are generally performed in both direction, which is why the
   translation machinery is largely agnostic to the distinction between target and source.

During these four passes, all changes are recorded in a specialized data structure that
then allows to filter items to be analyzed further and to render changes using the items'
source spans, ultimately leading to deterministic output. The implementation is found in
the `changes` submodule.

### Type checks implementation
Checking the types of a matching pair of items is one of the most important and most
complicated features of `rust-semverver`. Type checks are performed for type aliases,
constants, statics, ADT fields, and function signatures. This is implemented using the
type inference machinery of rustc and a custom `TypeComparisonContext`, located in
in the `typeck` module, that performs the necessary heavy lifting when given two types.

The general process is to translate one of the types to allow for comparison, and to use
inference variables for items that usually invoke inference mechanisms in the compiler,
like functions. Then, an equality check on the two types is performed, and a possible
error is lifted and registered in the change store. If such a type check succeeds without
error, bounds checks can be performed *in the same context*, even though they can be
performed without a type check where appropriate.

### Bounds checks implementation
Checking the bounds of a matching pair of items is performed for all items that are
subject to type changes, as well as trait definitions, using the already mentioned
`TypeComparisonContext`. The underlying mechanism is also used to match inherent, and --
to a lesser extend -- trait impls.

Bounds checks work in a similar manner to type checks. One of the items, in these case the
set of bounds, gets translated, and passed to an inference context. However, to properly
recognize all changes to trait bounds, this analysis step has to be performed in both
directions, to catch both loosening and tightening of bounds.

### Trait impl matching
All trait impls are matched up in both directions, to determine whether impls for specific
types have been added or removed (note that in this context, an impl refers to the
*existence of a trait implementation matching a given type, not a specific trait impl*. If
no match is found when checking an old impl, this implies that an impl has been removed,
and that it has been addded when a new impl has no old matching counterpart.

The actual matching is performed using the trait system. The translated bounds and trait
definition of the impl being checked are registered in a specialized `BoundContext`, which
is wrapping a fulfillment context that determines whether any matching impl exists.

### Inherent impl matching
Matching inherent impls roughly follows the same principles as checking trait impls.
However, there are a few vital differences, since different inherent impls of the same
type don't need to declare the same set of associated items. Thus, each associated item is
kept track of to determine the set of impls it is present in. Each of these impls needs to
be matched in the other crate, to find a matching associated item in each. Then, regular
type and structural checks are performed on the matching items.

The actual impl matching is performed based on the trait bounds on the inherent impls, as
described in a previous section.

## Tests
The change recording structure has a suite of unit tests to ensure correct behaviour with
regards to change categorization and storage, according to the usual convention, these
unit tests are located in the same file as the implementation. Various invariants are
tested using `quickcheck`, others are exercised as plain examples.

Most of the functionality, however, especially the analysis implementation, is tested
using an evergrowing integration test suite, which records the analysis results for mockup
crates, normalizes the output with regards to paths and similar information contained, and
compares it to a previously recorded version using `git`. Currently, regular crates are
supported in a limited fashion in this set of tests as well. However, to use this
functionality to the full extend, some changes to the compiler have yet to be upstreamed
at the time of writing.
