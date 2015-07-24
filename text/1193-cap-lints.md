- Feature Name: N/A
- Start Date: 2015-07-07
- RFC PR: [rust-lang/rfcs#1193](https://github.com/rust-lang/rfcs/pull/1193)
- Rust Issue: [rust-lang/rust#27259](https://github.com/rust-lang/rust/issues/27259)

# Summary

Add a new flag to the compiler, `--cap-lints`, which set the maximum possible
lint level for the entire crate (and cannot be overridden). Cargo will then pass
`--cap-lints allow` to all upstream dependencies when compiling code.

# Motivation

> Note: this RFC represents issue [#1029][issue]

Currently any modification to a lint in the compiler is strictly speaking a
breaking change. All crates are free to place `#![deny(warnings)]` at the top of
their crate, turning any new warnings into compilation errors. This means that
if a future version of Rust starts to emit new warnings it may fail to compile
some previously written code (a breaking change).

We would very much like to be able to modify lints, however. For example
[rust-lang/rust#26473][pr] updated the `missing_docs` lint to also look for
missing documentation on `const` items. This ended up [breaking some
crates][term-pr] in the ecosystem due to their usage of
`#![deny(missing_docs)]`.

[issue]: https://github.com/rust-lang/rfcs/issues/1029
[pr]: https://github.com/rust-lang/rust/pull/26473
[term-pr]: https://github.com/rust-lang/term/pull/34

The mechanism proposed in this RFC is aimed at providing a method to compile
upstream dependencies in a way such that they are resilient to changes in the
behavior of the standard lints in the compiler. A new lint warning or error will
never represent a memory safety issue (otherwise it'd be a real error) so it
should be safe to ignore any new instances of a warning that didn't show up
before.

# Detailed design

There are two primary changes propsed by this RFC, the first of which is a new
flag to the compiler:

```
    --cap-lints LEVEL   Set the maximum lint level for this compilation, cannot
                        be overridden by other flags or attributes.
```

For example when `--cap-lints allow` is passed, all instances of `#[warn]`,
`#[deny]`, and `#[forbid]` are ignored. If, however `--cap-lints warn` is passed
only `deny` and `forbid` directives are ignored.

The acceptable values for `LEVEL` will be `allow`, `warn`, `deny`, or `forbid`.

The second change proposed is to have Cargo pass `--cap-lints allow` to all
upstream dependencies. Cargo currently passes `-A warnings` to all upstream
dependencies (allow all warnings by default), so this would just be guaranteeing
that no lints could be fired for upstream dependencies.

With these two pieces combined together it is now possible to modify lints in
the compiler in a backwards compatible fashion. Modifications to existing lints
to emit new warnings will not get triggered, and new lints will also be entirely
suppressed **only for upstream dependencies**.

## Cargo Backwards Compatibility

This flag would be first non-1.0 flag that Cargo would be passing to the
compiler. This means that Cargo can no longer drive a 1.0 compiler, but only a
1.N+ compiler which has the `--cap-lints` flag. To handle this discrepancy Cargo
will detect whether `--cap-lints` is a valid flag to the compiler.

Cargo already runs `rustc -vV` to learn about the compiler (e.g. a "unique
string" that's opaque to Cargo) and it will instead start passing
`rustc -vV --cap-lints allow` to the compiler instead. This will allow Cargo to
simultaneously detect whether the flag is valid and learning about the version
string. If this command fails and `rustc -vV` succeeds then Cargo will fall back
to the old behavior of passing `-A warnings`.

# Drawbacks

This RFC adds surface area to the command line of the compiler with a relatively
obscure option `--cap-lints`. The option will almost never be passed by anything
other than Cargo, so having it show up here is a little unfortunate.

Some crates may inadvertently rely on memory safety through lints, or otherwise
very much not want lints to be turned off. For example if modifications to a new
lint to generate more warnings caused an upstream dependency to fail to compile,
it could represent a serious bug indicating the dependency needs to be updated.
This system would paper over this issue by forcing compilation to succeed. This
use case seems relatively rare, however, and lints are also perhaps not the best
method to ensure the safety of a crate.

Cargo may one day grow configuration to *not* pass this flag by default (e.g. go
back to passing `-Awarnings` by default), which is yet again more expansion of
API surface area.

# Alternatives

* Modifications to lints or additions to lints could be considered
  backwards-incompatible changes.
* The meaning of the `-A` flag could be reinterpreted as "this cannot be
  overridden"
* A new "meta lint" could be introduced to represent the maximum cap, for
  example `-A everything`. This is semantically different enough from `-A foo`
  that it seems worth having a new flag.

# Unresolved questions

None yet.
