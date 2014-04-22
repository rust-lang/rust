- Start Date: 2014-03-23
- RFC PR #:
- Rust Issue #:

# Summary

A proposal for making the `mod` keyword do what everybody expects, and make
it easier to manage multiple modules in a directory. The actual semantics of
the `use` keyword would not significantly change, but the practical use
would be tightened.

# Motivation

Note: in this proposal, I omit most use of the `pub` keyword, even if is
required. When I speak of "public", I refer only to conceptual visibility.

Frequently, newcomers to the language try to put a `mod` statement in
every file that accesses a module. Partilarly, they expect `mod` to work
like C/C++ `#include` or python `import`. C++ `using` or python `from x import`
only do a single duty of bringing a name into scope, but Rust's `use` does
the equivalent of both `#include` and `using`, and there is no equivalent
to Rust's `mod` at all.

Being confusing to newcomers is, in itself, a bug. But the strange purpose
of Rust's `mod` is also harmful even if you know what it does. Consider the
case when you have many files in a module. Several of the files in this
module require some shared routines. Currently, the mod.rs is required
to not only list the `mod` statements for the "public" modules (which is
reasonable, though some might argue in favor of `mod *;` to catch them all),
but also all of the private details that those modules use.

Apparently, the current way is not confusing if coming from Javascript,
but Javascript is an example of good *marketing*, not good *design*.

My proposal is mostly based on the way Python does it.

# Detailed design

## Source layout

- src/main.rs
  `mod fish;`
  `mod bird;`
  `mod penguin;`
- src/fish.rs
- src/bird.rs
  `mod penguin;`
- src/penguin/
 - src/penguin/mod.rs
   `mod foo;`
   `mod bar;`
   `use self::foo::popular_function;`
   `use super::bird;`
   `use ::fish::swim;`
 - src/penguin/foo.rs
   `mod bar;`
   `mod detail;`
 - src/penguin/bar.rs
   `mod detail;`
 - src/penguin/detail.rs
   `use self::bar;` (\*)
   `use super::fish;` (\*)

(Those (\*)s would need an extra super:: in alternative #3.)

## Stop

Go back and look at the source layout again.

I bet you (whether familiar with Rust, or only with other languages)
understood *exactly* what that source layout means without having to read
any documentation at all. Documentation is *never* a substitute for being
obvious.

## Suggested Approach

First, introduce the (transparent) concept of a 'package'. A package is a
collection of modules from one directory. A crate has a root package.
A package may contain subpackages, which are also modules.

One of the modules in a package is considered the root module. For the
root package, this is main.rs or lib.rs; for subpackages this is mod.rs.
All of the visible names in the root module are also visible when the
package is viewed as a module. Besides the obvious subpackage case, this
is important for locating `fn main`, and can also happen with `super::`.

To avoid cycles during compilation, the modules may be completely loaded,
or just registered to be loaded when done with the current load
(alternatively, the "return early" method may be used).

When actually loading a file, if the compiler see `mod foo` at any point -
even `mod bar { mod foo; }` - it tries to add foo from the current
directory. If foo/mod.rs exists, it is added as a new package (just like
the root package was), otherwise it is added as a module.

Note that any modules in a package that are not directly visible from its
root module will not be directly visible from the package when the package
is used as a module. With the example layout, `::penguin::detail` does not
exist (but `::penguin::foo::detail` and `::penguin::bar::detail` do - if they
didn't `pub` mod/use it, this has the benefits of Java's package-private).

# Alternatives

- Do nothing. Insist to the newcomers "it'll be less confusing later".
- Just eliminate `mod` entirely. Automatically find the right filename
    when `use` is used (this is basically how Java imports work).
- Fix the meaning of `mod`, but don't think about 'package'. This could
  work, but has a slightly pointless meaning for `self::` and `super::`

# Unresolved questions

All of these questions can be put off until after this RFC is implemented,
though due attention should be given to the first.

- Should `use foo::...` default to `use self::foo::...` or `use ::foo::...`
  I'm fond of defaulting `self::`, but rust currently does `::`.
  Perhaps it should be forbidden entirely (or gated) for a while?
- Are there any additional improvements that could be made for multiple
  crates? (make another RFC after this one is implemented).
- Should `mod *;` be added so a root module need not list all the .rs files?
- Should there be a way to not have a mod.rs at all?
- Should `mod foo { }` be feature-gated?
