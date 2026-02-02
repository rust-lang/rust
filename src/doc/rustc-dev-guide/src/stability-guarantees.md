# Stability guarantees

This page gives an overview of our stability guarantees.

## RFCs

* [RFC 1105 api evolution](https://github.com/rust-lang/rfcs/blob/master/text/1105-api-evolution.md)
* [RFC 1122 language semver](https://github.com/rust-lang/rfcs/blob/master/text/1122-language-semver.md)

## Blog posts

* [Stability as a Deliverable](https://blog.rust-lang.org/2014/10/30/Stability/)

## rustc-dev-guide links

* [Stabilizing library features](./stability.md)
* [Stabilizing language features](./stabilization_guide.md)
* [What qualifies as a bug fix?](./bug-fix-procedure.md#what-qualifies-as-a-bug-fix)

## Exemptions

Even if some of our infrastructure can be used by others, it is still considered
internal and comes without stability guarantees. This is a non-exhaustive list
of components without stability guarantees:

* The CLIs and environment variables used by `remote-test-client` / `remote-test-server`
