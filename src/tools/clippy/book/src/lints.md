# Clippy's Lints

Clippy offers a bunch of additional lints, to help its users write more correct
and idiomatic Rust code. A full list of all lints, that can be filtered by
category, lint level or keywords, can be found in the [Clippy lint
documentation].

This chapter will give an overview of the different lint categories, which kind
of lints they offer and recommended actions when you should see a lint out of
that category. For examples, see the [Clippy lint documentation] and filter by
category.

The different lint groups were defined in the [Clippy 1.0 RFC].

## Correctness

The `clippy::correctness` group is the only lint group in Clippy which lints are
deny-by-default and abort the compilation when triggered. This is for good
reason: If you see a `correctness` lint, it means that your code is outright
wrong or useless, and you should try to fix it.

Lints in this category are carefully picked and should be free of false
positives. So just `#[allow]`ing those lints is not recommended.

## Suspicious

The `clippy::suspicious` group is similar to the correctness lints in that it
contains lints that trigger on code that is really _sus_ and should be fixed. As
opposed to correctness lints, it might be possible that the linted code is
intentionally written like it is.

It is still recommended to fix code that is linted by lints out of this group
instead of `#[allow]`ing the lint. In case you intentionally have written code
that offends the lint you should specifically and locally `#[allow]` the lint
and add give a reason why the code is correct as written.

## Complexity

The `clippy::complexity` group offers lints that give you suggestions on how to
simplify your code. It mostly focuses on code that can be written in a shorter
and more readable way, while preserving the semantics.

If you should see a complexity lint, it usually means that you can remove or
replace some code, and it is recommended to do so. However, if you need the more
complex code for some expressiveness reason, it is recommended to allow
complexity lints on a case-by-case basis.

## Perf

The `clippy::perf` group gives you suggestions on how you can increase the
performance of your code. Those lints are mostly about code that the compiler
can't trivially optimize, but has to be written in a slightly different way to
make the optimizer job easier.

Perf lints are usually easy to apply, and it is recommended to do so.

## Style

The `clippy::style` group is mostly about writing idiomatic code. Because style
is subjective, this lint group is the most opinionated warn-by-default group in
Clippy.

If you see a style lint, applying the suggestion usually makes your code more
readable and idiomatic. But because we know that this is opinionated, feel free
to sprinkle `#[allow]`s for style lints in your code or `#![allow]` a style lint
on your whole crate if you disagree with the suggested style completely.

## Pedantic

The `clippy::pedantic` group makes Clippy even more _pedantic_. You can enable
the whole group with `#![warn(clippy::pedantic)]` in the `lib.rs`/`main.rs` of
your crate. This lint group is for Clippy power users that want an in depth
check of their code.

> _Note:_ Instead of enabling the whole group (like Clippy itself does), you may
> want to cherry-pick lints out of the pedantic group.

If you enable this group, expect to also use `#[allow]` attributes generously
throughout your code. Lints in this group are designed to be pedantic and false
positives sometimes are intentional in order to prevent false negatives.

## Restriction

The `clippy::restriction` group contains lints that will _restrict_ you from
using certain parts of the Rust language. It is **not** recommended to enable
the whole group, but rather cherry-pick lints that are useful for your code base
and your use case.

> _Note:_ Clippy will produce a warning if it finds a
> `#![warn(clippy::restriction)]` attribute in your code!

Lints from this group will restrict you in some way. If you enable a restriction
lint for your crate it is recommended to also fix code that this lint triggers
on. However, those lints are really strict by design, and you might want to
`#[allow]` them in some special cases, with a comment justifying that.

## Cargo

The `clippy::cargo` group gives you suggestions on how to improve your
`Cargo.toml` file. This might be especially interesting if you want to publish
your crate and are not sure if you have all useful information in your
`Cargo.toml`.

## Nursery

The `clippy::nursery` group contains lints which are buggy or need more work. It is **not** 
recommended to enable the whole group, but rather cherry-pick lints that are useful for your 
code base and your use case. 

## Deprecated

The `clippy::deprecated` is empty lints that exist to ensure that `#[allow(lintname)]` still 
compiles after the lint was deprecated. Deprecation "removes" lints by removing their 
functionality and marking them as deprecated, which may cause further warnings but cannot 
cause a compiler error.

[Clippy lint documentation]: https://rust-lang.github.io/rust-clippy/
[Clippy 1.0 RFC]: https://github.com/rust-lang/rfcs/blob/master/text/2476-clippy-uno.md#lint-audit-and-categories
