- Feature Name: macros-literal-match
- Start Date: 2016-04-08
- RFC PR: https://github.com/rust-lang/rfcs/pull/1576
- Rust Issue: https://github.com/rust-lang/rust/issues/35625

# Summary

Add a `literal` fragment specifier for `macro_rules!` patterns that matches literal constants:

```rust
macro_rules! foo {
    ($l:literal) => ( /* ... */ );
};
```

# Motivation

There are a lot of macros out there that take literal constants as arguments (often string constants). For now, most use the `expr` fragment specifier, which is fine since literal constants are a subset of expressions. But it has the following issues:
* It restricts the syntax of those macros. A limited set of FOLLOW tokens is allowed after an `expr` specifier. For example `$e:expr : $t:ty` is not allowed whereas `$l:literal : $t:ty` should be. There is no reason to arbitrarily restrict the syntax of those macros where they will only be actually used with literal constants. A workaround for that is to use the `tt` matcher.
* It does not allow for proper error reporting where the macro actually *needs* the parameter to be a literal constant. With this RFC, bad usage of such macros will give a proper syntax error message whereas with `epxr` it would probably give a syntax or typing error inside the generated code, which is hard to understand.
* It's not consistent. There is no reason to allow expressions, types, etc. but not literals.

# Design

Add a `literal` (or `lit`, or `constant`) matcher in macro patterns that matches all single-tokens literal constants (those that are currently represented by `token::Literal`).
Matching input against this matcher would call the `parse_lit` method from `libsyntax::parse::Parser`. The FOLLOW set of this matcher should be the same as `ident` since it matches a single token.

# Drawbacks

This includes only single-token literal constants and not compound literals, for example struct literals `Foo { x: some_literal, y: some_literal }` or arrays `[some_literal ; N]`, where `some_literal` can itself be a compound literal. See in alternatives why this is disallowed.

# Alternatives

* Allow compound literals too. In theory there is no reason to exclude them since they do not require any computation. In practice though, allowing them requires using the expression parser but limiting it to allow only other compound literals and not arbitrary expressions to occur inside a compound literal (for example inside struct fields). This would probably require much more work to implement and also mitigates the first motivation since it will probably restrict a lot the FOLLOW set of such fragments.
* Adding fragment specifiers for each constant type: `$s:str` which expects a literal string, `$i:integer` which expects a literal integer, etc. With this design, we could allow something like `$s:struct` for compound literals which still requires a lot of work to implement but has the advantage of not ‶polluting″ the FOLLOW sets of other specifiers such as `str`. It provides also better ‶static″ (pre-expansion) checking of the arguments of a macro and thus better error reporting. Types are also good for documentation. The main drawback here if of course that we could not allow any possible type since we cannot interleave parsing and type checking, so we would have to define a list of accepted types, for example `str`, `integer`, `bool`, `struct` and `array` (without specifying the complete type of the structs and arrays). This would be a bit inconsistent since those types indeed refer more to syntactic categories in this context than to true Rust types. It would be frustrating and confusing since it can give the impression that macros do type-checking of their arguments, when of course they don't.
* Don't do this. Continue to use `expr` or `tt` to refer to literal constants.

# Unresolved

The keyword of the matcher can be `literal`, `lit`, `constant`, or something else.
