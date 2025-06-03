# Fuzzing

<!-- date-check: Mar 2023 -->

For the purposes of this guide, *fuzzing* is any testing methodology that
involves compiling a wide variety of programs in an attempt to uncover bugs in
rustc. Fuzzing is often used to find internal compiler errors (ICEs). Fuzzing
can be beneficial, because it can find bugs before users run into them and
provide small, self-contained programs that make the bug easier to track down.
However, some common mistakes can reduce the helpfulness of fuzzing and end up
making contributors' lives harder. To maximize your positive impact on the Rust
project, please read this guide before reporting fuzzer-generated bugs!

## Guidelines

### In a nutshell

*Please do:*

- Ensure the bug is still present on the latest nightly rustc
- Include a reasonably minimal, standalone example along with any bug report
- Include all of the information requested in the bug report template
- Search for existing reports with the same message and query stack
- Format the test case with `rustfmt`, if it maintains the bug
- Indicate that the bug was found by fuzzing

*Please don't:*

- Don't report lots of bugs that use internal features, including but not
  limited to `custom_mir`, `lang_items`, `no_core`, and `rustc_attrs`.
- Don't seed your fuzzer with inputs that are known to crash rustc (details
  below).

### Discussion

If you're not sure whether or not an ICE is a duplicate of one that's already
been reported, please go ahead and report it and link to issues you think might
be related. In general, ICEs on the same line but with different *query stacks*
are usually distinct bugs. For example, [#109020][#109020] and [#109129][#109129]
had similar error messages:

```
error: internal compiler error: compiler/rustc_middle/src/ty/normalize_erasing_regions.rs:195:90: Failed to normalize <[closure@src/main.rs:36:25: 36:28] as std::ops::FnOnce<(Emplacable<()>,)>>::Output, maybe try to call `try_normalize_erasing_regions` instead
```
```
error: internal compiler error: compiler/rustc_middle/src/ty/normalize_erasing_regions.rs:195:90: Failed to normalize <() as Project>::Assoc, maybe try to call `try_normalize_erasing_regions` instead
```
but different query stacks:
```
query stack during panic:
#0 [fn_abi_of_instance] computing call ABI of `<[closure@src/main.rs:36:25: 36:28] as core::ops::function::FnOnce<(Emplacable<()>,)>>::call_once - shim(vtable)`
end of query stack
```
```
query stack during panic:
#0 [check_mod_attrs] checking attributes in top-level module
#1 [analysis] running analysis passes on this crate
end of query stack
```

[#109020]: https://github.com/rust-lang/rust/issues/109020
[#109129]: https://github.com/rust-lang/rust/issues/109129

## Building a corpus

When building a corpus, be sure to avoid collecting tests that are already
known to crash rustc. A fuzzer that is seeded with such tests is more likely to
generate bugs with the same root cause, wasting everyone's time. The simplest
way to avoid this is to loop over each file in the corpus, see if it causes an
ICE, and remove it if so.

To build a corpus, you may want to use:

- The rustc/rust-analyzer/clippy test suites (or even source code) --- though avoid
  tests that are already known to cause failures, which often begin with comments
  like `//@ failure-status: 101` or `//@ known-bug: #NNN`.
- The already-fixed ICEs in the archived [Glacier][glacier] repository --- though
  avoid the unfixed ones in `ices/`!

[glacier]: https://github.com/rust-lang/glacier

## Extra credit

Here are a few things you can do to help the Rust project after filing an ICE.

- [Bisect][bisect] the bug to figure out when it was introduced.
  If you find the regressing PR / commit, you can mark the issue with the label
  `S-has-bisection`. If not, consider applying `E-needs-bisection` instead.
- Fix "distractions": problems with the test case that don't contribute to
  triggering the ICE, such as syntax errors or borrow-checking errors
- Minimize the test case (see below). If successful, you can label the
  issue with `S-has-mcve`. Otherwise, you can apply `E-needs-mcve`.
- Add the minimal test case to the rust-lang/rust repo as a [crashes test].
  While you're at it, consider including other "untracked" crashes in your PR.
  Please don't forget to mark your issue with `S-bug-has-test` afterwards.

See also [applying and removing labels][labeling].

[bisect]: https://rust-lang.github.io/cargo-bisect-rustc/
[crashes test]: tests/compiletest.html#crashes-tests
[labeling]: https://forge.rust-lang.org/release/issue-triaging.html#applying-and-removing-labels

## Minimization

It is helpful to carefully *minimize* the fuzzer-generated input. When
minimizing, be careful to preserve the original error, and avoid introducing
distracting problems such as syntax, type-checking, or borrow-checking errors.

There are some tools that can help with minimization. If you're not sure how
to avoid introducing syntax, type-, and borrow-checking errors while using
these tools, post both the complete and minimized test cases. Generally,
*syntax-aware* tools give the best results in the least amount of time.
[`treereduce-rust`][treereduce] and [picireny][picireny] are syntax-aware.
[`halfempty`][halfempty] is not, but is generally a high-quality tool.

[halfempty]: https://github.com/googleprojectzero/halfempty
[picireny]: https://github.com/renatahodovan/picireny
[treereduce]: https://github.com/langston-barrett/treereduce

## Effective fuzzing

When fuzzing rustc, you may want to avoid generating machine code, since this
is mostly done by LLVM. Try `--emit=mir` instead.

A variety of compiler flags can uncover different issues. `-Zmir-opt-level=4`
will turn on MIR optimization passes that are not run by default, potentially
uncovering interesting bugs. `-Zvalidate-mir` can help uncover such bugs.

If you're fuzzing a compiler you built, you may want to build it with `-C
target-cpu=native` or even PGO/BOLT to squeeze out a few more executions per
second. Of course, it's best to try multiple build configurations and see
what actually results in superior throughput.

You may want to build rustc from source with debug assertions to find
additional bugs, though this is a trade-off: it can slow down fuzzing by
requiring extra work for every execution. To enable debug assertions, add this
to `bootstrap.toml` when compiling rustc:

```toml
[rust]
debug-assertions = true
```

ICEs that require debug assertions to reproduce should be tagged 
[`requires-debug-assertions`][requires-debug-assertions].

[requires-debug-assertions]: https://github.com/rust-lang/rust/labels/requires-debug-assertions

## Existing projects

- [fuzz-rustc][fuzz-rustc] demonstrates how to fuzz rustc with libfuzzer
- [icemaker][icemaker] runs rustc and other tools on a large number of source
  files with a variety of flags to catch ICEs
- [tree-splicer][tree-splicer] generates new source files by combining existing
  ones while maintaining correct syntax

[fuzz-rustc]: https://github.com/dwrensha/fuzz-rustc
[icemaker]: https://github.com/matthiaskrgr/icemaker/
[tree-splicer]: https://github.com/langston-barrett/tree-splicer/
