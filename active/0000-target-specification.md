- Start Date: 2014-06-18
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

*Note:* This RFC discusses the behavior of `rustc`, and not any changes to the
language.

Change how target specification is done to be more flexible for unexpected
usecases. Additionally, add support for the "unknown" OS in target triples,
providing a minimum set of target specifications that is valid for bare-metal
situations.

# Motivation

One of Rust's important use cases is embedded, OS, or otherwise "bare metal"
software. At the moment, we still depend on LLVM's split-stack prologue for
stack safety. In certain situations, it is impossible or undesirable to
support what LLVM requires to enable this (on x86, a certain thread-local
storage setup).

# Detailed design

A target triple consists of three strings separated by a hyphen, with a
possible fourth string at the end preceded by a hyphen. The first is the
architecture, the second is the "vendor", the third is the OS type, and the
optional fourth is environment type. In theory, this specifies precisely what
platform the generated binary will be able to run on. All of this is
determined not by us but by LLVM and other tools. When on bare metal or a
similar environment, there essentially is no OS, and to handle this there is
the concept of "unknown" in the target triple.  When the OS is "unknown",
no runtime environment is assumed to be present (including things such as
dynamic linking, threads/thread-local storage, IO, etc).

Rather than listing specific targets for special treatment, introduce a
general mechanism for specifying certain characteristics of a target triple.
Redesign how targets are handled around this specification, including for the
built-in targets. Add a `T` family of flags, similar to `C`, for target
specification, as well as `-T from-file=targ.json` which will load the
configuration from a JSON file. A table of the flags and their meaning:

* `data-layout`: The [LLVM data
layout](http://llvm.org/docs/LangRef.html#data-layout) to use. Mostly included
for completeness; changing this is unlikely to be used.
* `link-args`: Arguments to pass to the linker, unconditionally.
* `cpu`: Default CPU to use for the target, overridable with `-C target-cpu`
* `features`: Default target features to enable, augmentable with `-C
  target-features`.
* `dynamic-linking-available`: Whether the `dylib` crate type is allowed.
* `split-stacks-supported`: Whether there is runtime support that will allow
  LLVM's split stack prologue to function as intended.
* `target-name`: What name to use for `targ_os` for use in `cfg`.

Rather than hardcoding a specific set of behaviors per-target, with no
recourse for escaping them, the compiler would also use this mechanism when
deciding how to build for a given target. The process would look like:

1. Look up the target triple in an internal map, and load that configuration
   if it exists.
2. If `-T from-file` is given, load any options from that file.
3. For every other `-T` flag, let it override both of the above.
4. If `-C target-cpu` is specified, replace the `cpu` with it.
5. If `-C features` is specified, add those to the ones specified by `-T`.

Then during compilation, this information is used at the proper places rather
than matching against an enum listing the OSes we recognize.

# Drawbacks

More complexity. However, this is very flexible and allows one to use Rust on
a new or non-standard target *incredibly easy*, without having to modify the
compiler. rustc is the only compiler I know of that would allow that.

# Alternatives

One possible extension is to load `<target>.json` from some directory, rather
than having to require `-T from-file`.

A less holistic approach would be to just allow disabling split stacks on a
per-crate basis. Another solution could be adding a family of targets,
`<arch>-unknown-unknown`, which omits all of the above complexity but does not
allow extending to new targets easily. `-T` can easily be extended for the
future needs of other targets.
