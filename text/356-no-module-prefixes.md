- Start Date: 2014-10-15
- RFC PR: [rust-lang/rfcs#356](https://github.com/rust-lang/rfcs/pull/356)
- Rust Issue: [rust-lang/rust#18073](https://github.com/rust-lang/rust/issues/18073)

# Summary

This is a conventions RFC that proposes that the items exported from a module
should *never* be prefixed with that module name. For example, we should have
`io::Error`, not `io::IoError`.

(An alternative design is included that special-cases overlap with the
`prelude`.)

# Motivation

Currently there is no clear prohibition around including the module's name as a
prefix on an exported item, and it is sometimes done for type names that are
feared to be "popular" (like `Error` and `Result` being `IoError` and
`IoResult`) for clarity.

This RFC include two designs: one that entirely rules out such prefixes, and one
that rules it out *except* for names that overlap with the prelude. Pros/cons
are given for each.

# Detailed design

The main rule being proposed is very simple: the items exported from a module
should never be prefixed with the module's name.

Rationale:

* Avoids needless stuttering like `io::IoError`.
* Any ambiguity can be worked around:
    * Either qualify by the module, i.e. `io::Error`,
    * Or rename on import: `use io::Error as IoError`.
* The rule is extremely simple and clear.

Downsides:

* The name may already exist in the module wanting to export it.
    * If that's due to explicit imports, those imports can be renamed or
      module-qualified (see above).
    * If that's due to a *prelude* conflict, however, confusion may arise due to
      the conventional *global* meaning of identifiers defined in the prelude
      (i.e., programmers do not expect prelude imports to be shadowed).

Overall, the RFC author believes that *if* this convention is adopted, confusion
around redefining prelude names would gradually go away, because (at least for
things like `Result`) we would come to expect it.

# Alternative design

An alternative rule would be to never prefix an exported item with the module's
name, *except* for names that are also defined in the prelude, which *must* be
prefixed by the module's name.

For example, we would have `io::Error` and `io::IoResult`.

Rationale:

* Largely the same as the above, but less decisively.
* Avoids confusion around prelude-defined names.

Downsides:

* Retains stuttering for some important cases, e.g. custom `Result` types, which
  are likely to be fairly common.
* Makes it even more problematic to expand the prelude in the future.
