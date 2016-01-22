- Start Date: 2014-06-18
- RFC PR: [rust-lang/rfcs#131](https://github.com/rust-lang/rfcs/pull/131)
- Rust Issue: [rust-lang/rust#16093](https://github.com/rust-lang/rust/issues/16093)

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
storage setup). Additionally, porting `rustc` to a new platform requires
modifying the compiler, adding a new OS manually.

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
built-in targets. Extend the `--target` flag to accept a file name of a target
specification. A table of the target specification flags and their meaning:

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
* `llvm-target`: What target to pass to LLVM.
* `relocation-model`: What relocation model to use by default.
* `target_endian`, `target_word_size`: Specify the strings used for the
  corresponding `cfg` variables.
* `code-model`: Code model to pass to LLVM, overridable with `-C code-model`.
* `no-redzone`: Disable use of any stack redzone, overridable with `-C
  no-redzone`

Rather than hardcoding a specific set of behaviors per-target, with no
recourse for escaping them, the compiler would also use this mechanism when
deciding how to build for a given target. The process would look like:

1. Look up the target triple in an internal map, and load that configuration
   if it exists. If that fails, check if the target name exists as a file, and
   try loading that. If the file does not exist, look up `<target>.json` in
   the `RUST_TARGET_PATH`, which is a colon-separated list of directories.
2. If `-C linker` is specified, use that instead of the target-specified
   linker.
3. If `-C link-args` is given, add those to the ones specified by the target.
4. If `-C target-cpu` is specified, replace the target `cpu` with it.
5. If `-C target-feature` is specified, add those to the ones specified by the
   target.
6. If `-C relocation-model` is specified, replace the target
   `relocation-model` with it.
7. If `-C code-model` is specified, replace the target `code-model` with it.
8. If `-C no-redzone` is specified, replace the target `no-redzone` with true.


Then during compilation, this information is used at the proper places rather
than matching against an enum listing the OSes we recognize. The `target_os`,
`target_family`, and `target_arch` `cfg` variables would be extracted from the
`--target` passed to rustc.

# Drawbacks

More complexity. However, this is very flexible and allows one to use Rust on
a new or non-standard target *incredibly easy*, without having to modify the
compiler. rustc is the only compiler I know of that would allow that.

# Alternatives

A less holistic approach would be to just allow disabling split stacks on a
per-crate basis. Another solution could be adding a family of targets,
`<arch>-unknown-unknown`, which omits all of the above complexity but does not
allow extending to new targets easily.
