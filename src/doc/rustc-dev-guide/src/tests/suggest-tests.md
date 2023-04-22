# Suggest tests tool

This chapter is about the internals of and contribution instructions for the
`suggest-tests` tool. For a high-level overview of the tool, see
[this section](../building/suggested.md#x-suggest). This tool is currently in a
beta state and is tracked by [this](https://github.com/rust-lang/rust/issues/109933)
issue on Github. Currently the number of tests it will suggest are very limited
in scope, we are looking to expand this (contributions welcome!).

## Internals

The tool is defined in a separate crate ([`src/tools/suggest-tests`](https://github.com/rust-lang/rust/blob/master/src/tools/suggest-tests))
which outputs suggestions which are parsed by a shim in bootstrap
([`src/bootstrap/suggest.rs`](https://github.com/rust-lang/rust/blob/master/src/bootstrap/suggest.rs)).
The only notable thing the bootstrap shim does is (when invoked with the
`--run` flag) use bootstrap's internal mechanisms to create a new `Builder` and
uses it to invoke the suggested commands. The `suggest-tests` crate is where the
fun happens, two kinds of suggestions are defined: "static" and "dynamic"
suggestions.

### Static suggestions

Defined [here](https://github.com/rust-lang/rust/blob/master/src/tools/suggest-tests/src/static_suggestions.rs).
Static suggestions are simple: they are just [globs](https://crates.io/crates/glob)
which map to a `x` command. In `suggest-tests`, this is implemented with a
simple `macro_rules` macro.

### Dynamic suggestions

Defined [here](https://github.com/rust-lang/rust/blob/master/src/tools/suggest-tests/src/dynamic_suggestions.rs).
These are more complicated than static suggestions and are implemented as
functions with the following signature: `fn(&Path) -> Vec<Suggestion>`. In
other words, each suggestion takes a path to a modified file and (after running
arbitrary Rust code) can return any number of suggestions, or none. Dynamic
suggestions are useful for situations where fine-grained control over
suggestions is needed. For example, modifications to the `compiler/xyz/` path
should trigger the `x test compiler/xyz` suggestion. In the future, dynamic
suggestions might even read file contents to determine if (what) tests should
run.

## Adding a suggestion

The following steps should serve as a rough guide to add suggestions to
`suggest-tests` (very welcome!):

1. Determine the rules for your suggestion. Is it simple and operates only on
   a single path or does it match globs? Does it need fine-grained control over
   the resulting command or does "one size fit all"?
2. Based on the previous step, decide if your suggestion should be implemented
   as either static or dynamic.
3. Implement the suggestion. If it is dynamic then a test is highly recommended,
   to verify that your logic is correct and to give an example of the suggestion.
   See the [tests.rs](https://github.com/rust-lang/rust/blob/master/src/tools/suggest-tests/src/tests.rs)
   file.
4. Open a PR implementing your suggestion. **(TODO: add example PR)**
