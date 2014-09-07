- Start Date: 2014-05-21
- RFC PR: [rust-lang/rfcs#85](https://github.com/rust-lang/rfcs/pull/85)
- Rust Issue: [rust-lang/rust#14473](https://github.com/rust-lang/rust/issues/14473)

# Summary

Allow macro expansion in patterns, i.e.

~~~ .rs
match x {
    my_macro!() => 1,
    _ => 2,
}
~~~

# Motivation

This is consistent with allowing macros in expressions etc.  It's also a year-old [open issue](https://github.com/mozilla/rust/issues/6830).

I have [implemented](https://github.com/mozilla/rust/pull/14298) this feature already and I'm [using it](https://github.com/kmcallister/html5/blob/937684f107090741c8e87135efc6e5476489857b/src/tree_builder/mod.rs#L111-L117) to [condense](https://github.com/kmcallister/html5/blob/937684f107090741c8e87135efc6e5476489857b/src/tree_builder/mod.rs#L261-L269) some ubiquitous patterns in the [HTML parser](https://github.com/kmcallister/html5) I'm writing.  This makes the code more concise and easier to cross-reference with the spec.

# Drawbacks / alternatives

A macro invocation in this position:

~~~ .rs
match x {
    my_macro!()
~~~

could potentially expand to any of three different syntactic elements:

* A pattern, i.e. `Foo(x)`
* The left side of a `match` arm, i.e. `Foo(x) | Bar(x) if x > 5`
* An entire `match` arm, i.e. `Foo(x) | Bar(x) if x > 5 => 1`

This RFC proposes only the first of these, but the others would be more useful in some cases.  Supporting multiple of the above would be significantly more complex.

Another alternative is to use a macro for the entire `match` expression, e.g.

~~~ .rs
my_match!(x {
    my_new_syntax => 1,
    _ => 2,
})
~~~

This doesn't involve any language changes, but requires writing a complicated procedural macro.  (My sustained attempts to do things like this with MBE macros have all failed.)  Perhaps I could alleviate some of the pain with a library for writing `match`-like macros, or better use of the existing parser in `libsyntax`.

The `my_match!` approach is also not very composable.

Another small drawback: `rustdoc` [can't document](https://github.com/kmcallister/rust/blob/af65e3e9824087a472de3fea3c7cb1efcec4550b/src/librustdoc/clean.rs#L1287-L1291) the name of a function argument which is produced by a pattern macro.

# Unresolved questions

None, as far as I know.
