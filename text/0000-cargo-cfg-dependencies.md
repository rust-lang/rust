- Feature Name: N/A
- Start Date: 2015-11-10
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Improve the target-specific dependency experience in Cargo by leveraging the
same `#[cfg]` syntax that Rust has.

# Motivation
[motivation]: #motivation

Currently in Cargo it's [relatively painful][issue] to list target-specific
dependencies. This can only be done by listing out the entire target string as
opposed to using the more-convenient `#[cfg]` annotations that Rust source code
has access to. Consequently a Windows-specific dependency ends up having to be
defined for four triples: `{i686,x86_64}-pc-windows-{gnu,msvc}`, and this is
unfortunately not forwards compatible as well!

[issue]: https://github.com/rust-lang/cargo/issues/1007

As a result most crates end up unconditionally depending on target-specific
dependencies and rely on the crates themselves to have the relevant `#[cfg]` to
only be compiled for the right platforms. This experience leads to excessive
downloads, excessive compilations, and overall "unclean methods" to have a
platform specific dependency.

This RFC proposes leveraging the same familiar syntax used in Rust itself to
define these dependencies.

# Detailed design
[design]: #detailed-design

The target-specific dependency syntax in Cargo will be expanded to include
not only full target strings but also `#[cfg]` expressions:

```toml
[target."cfg(windows)".dependencies]
winapi = "0.2"

[target."cfg(unix)".dependencies]
unix-socket = "0.4"

[target."cfg(target_os = \"macos\")".dependencies]
core-foundation = "0.2"
```

Specifically, the "target" listed here is considered special if it starts with
the string "cfg(" and ends with ")". If this is not true then Cargo will
continue to treat it as an opaque string and pass it to the compiler via
`--target` (Cargo's current behavior).

> **Note**: There's an [issue open against TOML][toml-issue] to support
> single-quoted keys allowing more ergonomic syntax in some cases like:
>
> ```toml
> [target.'cfg(target_os = "macos")'.dependencies]
> core-foundation = "0.2"
> ```

[toml-issue]: https://github.com/toml-lang/toml/issues/354

Cargo will implement its own parser of this syntax inside the `cfg` expression,
it will not rely on the compiler itself. The grammar, however, will be the same
as the compiler for now:

```
cfg := "cfg(" meta-item * ")"
meta-item := ident |
             ident "=" string |
             ident "(" meta-item * ")"
```

Like Rust, Cargo will implement the `any`, `all`, and `not` operators for the
`ident(list)` syntax. The last missing piece is simply understand what `ident`
and `ident = "string"` values are defined for a particular target. To learn this
information Cargo will query the compiler via a new command line flag:

```
$ rustc --print cfg
unix
target_os="apple"
target_pointer_width="64"
...

$ rustc --print cfg --target i686-pc-windows-msvc
windows
target_os="windows"
target_pointer_width="32"
...
```

The `--print cfg` command line flag will print out all built-in `#[cfg]`
directives defined by the compiler onto standard output. Each cfg will be
printed on its own line to allow external parsing. Cargo will use this to call
the compiler once (or twice if an explicit target is requested) when resolution
starts, and it will use these key/value pairs to execute the `cfg` queries in
the dependency graph being constructed.

# Drawbacks
[drawbacks]: #drawbacks

This is not a forwards-compatible extension to Cargo, so this will break
compatibility with older Cargo versions. If a crate is published with a Cargo
that supports this `cfg` syntax, it will not be buildable by a Cargo that does
not understand the `cfg` syntax. The registry itself is prepared to handle this
sort of situation as the "target" string is just opaque, however.

This can be perhaps mitigated via a number of strategies:

1. Have crates.io reject the `cfg` syntax until the implementation has landed on
   stable Cargo for at least one full cycle. Applications, path dependencies,
   and git dependencies would still be able to use this syntax, but crates.io
   wouldn't be able to leverage it immediately.
2. Crates on crates.io wishing for compatibility could simply hold off on using
   this syntax until this implementation has landed in stable Cargo for at least
   a full cycle. This would mean that everyone could use it immediately but "big
   crates" would be advised to hold off for compatibility for awhile.
3. Have crates.io rewrite dependencies as they're published. If you publish a
   crate with a `cfg(windows)` dependency then crates.io could expand this to
   all known triples which match `cfg(windows)` when storing the metadata
   internally. This would mean that crates using `cfg` syntax would continue to
   be compatible with older versions of Cargo so long as they were only used as
   a crates.io dependency.

For ease of implementation this RFC would recommend strategy (1) to help ease
this into the ecosystem without too much pain in terms of compatibility or
implementation.

# Alternatives
[alternatives]: #alternatives

Instead of using Rust's `#[cfg]` syntax, Cargo could support other options such
as patterns over the target string. For example it could accept something along
the lines of:

```toml
[target."*-pc-windows-*".dependencies]
winapi = "0.2"

[target."*-apple-*".dependencies]
core-foundation = "0.2"
```

While certainly more flexible than today's implementation, it unfortunately is
relatively error prone and doesn't cover all the use cases one may want:

* Matching against a string isn't necessarily guaranteed to be robust moving
  forward into the future.
* This doesn't support negation and other operators, e.g. `all(unix, not(osx))`.
* This doesn't support meta-families like `cfg(unix)`.

Another possible alternative would be to have Cargo supply pre-defined families
such as `windows` and `unix` as well as the above pattern matching, but this
eventually just moves into the territory of what `#[cfg]` already provides but
may not always quite get there.

# Unresolved questions
[unresolved]: #unresolved-questions

* This is not the only change that's known to Cargo which is known to not be
  forwards-compatible, so it may be best to lump them all together into one
  Cargo release instead of releasing them over time, but should this be blocked
  on those ideas? (note they have not been formed into an RFC yet)


