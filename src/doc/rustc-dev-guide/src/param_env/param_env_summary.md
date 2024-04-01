# The `ParamEnv` type

## Summary

The [`ParamEnv`][pe] is used to store information about the environment that we are interacting with the type system from. For example the set of in-scope where-clauses is stored in `ParamEnv` as it differs between each item whereas the list of user written impls is not stored in the `ParamEnv` as this does not change for each item.

This chapter of the dev guide covers:
- A high level summary of what a `ParamEnv` is and what it is used for
- Technical details about what the process of constructing a `ParamEnv` involves
- Guidance about how to acquire a `ParamEnv` when one is required

## Bundling

A useful API on `ParamEnv` is the [`and`][and] method which allows bundling a value with the `ParamEnv`. The `and` method produces a [`ParamEnvAnd<T>`][pea] making it clearer that using the inner value is intended to be done in that specific environment.

[and]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.and
[pe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html
[pea]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnvAnd.html