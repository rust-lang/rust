% The Rust Design FAQ

This document describes decisions that were arrived at after lengthy discussion and
experimenting with alternatives. Please do not propose reversing them unless
you have a new, extremely compelling argument. Note that this document
specifically talks about the *language* and not any library or implementation.

A few general guidelines define the philosophy:

- [Memory safety][mem] must never be compromised
- [Abstraction][abs] should be zero-cost, while still maintaining safety
- Practicality is key

[mem]: http://en.wikipedia.org/wiki/Memory_safety
[abs]: http://en.wikipedia.org/wiki/Abstraction_%28computer_science%29

# Semantics

## Data layout is unspecified

In the general case, `enum` and `struct` layout is undefined. This allows the
compiler to potentially do optimizations like re-using padding for the
discriminant, compacting variants of nested enums, reordering fields to remove
padding, etc. `enum`s which carry no data ("C-like") are eligible to have a
defined representation. Such `enum`s are easily distinguished in that they are
simply a list of names that carry no data:

```
enum CLike {
    A,
    B = 32,
    C = 34,
    D
}
```

The [repr attribute][repr] can be applied to such `enum`s to give them the same
representation as a primitive. This allows using Rust `enum`s in FFI where C
`enum`s are also used, for most use cases. The attribute can also be applied
to `struct`s to get the same layout as a C struct would.

[repr]: reference.html#miscellaneous-attributes

## There is no GC

A language that requires a GC is a language that opts into a larger, more
complex runtime than Rust cares for. Rust is usable on bare metal with no
extra runtime. Additionally, garbage collection is frequently a source of
non-deterministic behavior. Rust provides the tools to make using a GC
possible and even pleasant, but it should not be a requirement for
implementing the language.

## Non-`Sync` `static mut` is unsafe

Types which are [`Sync`][sync] are thread-safe when multiple shared
references to them are used concurrently. Types which are not `Sync` are not
thread-safe, and thus when used in a global require unsafe code to use.

[sync]: core/kinds/trait.Sync.html

### If mutable static items that implement `Sync` are safe, why is taking &mut SHARABLE unsafe?

Having multiple aliasing `&mut T`s is never allowed. Due to the nature of
globals, the borrow checker cannot possibly ensure that a static obeys the
borrowing rules, so taking a mutable reference to a static is always unsafe.

## There is no life before or after main (no static ctors/dtors)

Globals can not have a non-constant-expression constructor and cannot have a
destructor at all. This is an opinion of the language. Static constructors are
undesirable because they can slow down program startup. Life before main is
often considered a misfeature, never to be used. Rust helps this along by just
not having the feature.

See [the C++ FQA][fqa]  about the "static initialization order fiasco", and
[Eric Lippert's blog][elp] for the challenges in C#, which also has this
feature.

A nice replacement is the [lazy constructor macro][lcm] by [Marvin
Löbel][kim].

[fqa]: https://mail.mozilla.org/pipermail/rust-dev/2013-April/003815.html
[elp]: http://ericlippert.com/2013/02/06/static-constructors-part-one/
[lcm]: https://gist.github.com/Kimundi/8782487
[kim]: https://github.com/Kimundi

## The language does not require a runtime

See the above entry on GC. Requiring a runtime limits the utility of the
language, and makes it undeserving of the title "systems language". All Rust
code should need to run is a stack.

## `match` must be exhaustive

`match` being exhaustive has some useful properties. First, if every
possibility is covered by the `match`, adding further variants to the `enum`
in the future will prompt a compilation failure, rather than runtime failure.
Second, it makes cost explicit. In general, only safe way to have a
non-exhaustive match would be to fail the task if nothing is matched, though
it could fall through if the type of the `match` expression is `()`. This sort
of hidden cost and special casing is against the language's philosophy. It's
easy to ignore certain cases by using the `_` wildcard:

```rust,ignore
match val.do_something() {
    Cat(a) => { /* ... */ }
    _      => { /* ... */ }
}
```

[#3101][iss] is the issue that proposed making this the only behavior, with
rationale and discussion.

[iss]: https://github.com/rust-lang/rust/issues/3101

## No guaranteed tail-call optimization

In general, tail-call optimization is not guaranteed: see [here][tml] for a
detailed explanation with references. There is a [proposed extension][tce] that
would allow tail-call elimination in certain contexts. The compiler is still
free to optimize tail-calls [when it pleases][sco], however.

[tml]: https://mail.mozilla.org/pipermail/rust-dev/2013-April/003557.html
[sco]: http://llvm.org/docs/CodeGenerator.html#sibling-call-optimization
[tce]: https://github.com/rust-lang/rfcs/pull/81

## No constructors

Functions can serve the same purpose as constructors without adding any
language complexity.

## No copy constructors

Types which implement [`Copy`][copy], will do a standard C-like "shallow copy"
with no extra work (similar to "plain old data" in C++). It is impossible to
implement `Copy` types that require custom copy behavior. Instead, in Rust
"copy constructors" are created by implementing the [`Clone`][clone] trait,
and explicitly calling the `clone` method. Making user-defined copy operators
explicit surfaces the underlying complexity, forcing the developer to opt-in
to potentially expensive operations.

[copy]: core/kinds/trait.Copy.html
[clone]: core/clone/trait.Clone.html

## No move constructors

Values of all types are moved via `memcpy`. This makes writing generic unsafe
code much simpler since assignment, passing and returning are known to never
have a side effect like unwinding.

# Syntax

## Macros require balanced delimiters

This is to make the language easier to parse for machines. Since the body of a
macro can contain arbitrary tokens, some restriction is needed to allow simple
non-macro-expanding lexers and parsers. This comes in the form of requiring
that all delimiters be balanced.

## `->` for function return type

This is to make the language easier to parse for humans, especially in the face
of higher-order functions. `fn foo<T>(f: fn(int): int, fn(T): U): U` is not
particularly easy to read.

## `let` is used to introduce variables

`let` not only defines variables, but can do pattern matching. One can also
redeclare immutable variables with `let`. This is useful to avoid unnecessary
`mut` annotations. An interesting historical note is that Rust comes,
syntactically, most closely from ML, which also uses `let` to introduce
bindings.

See also [a long thread][alt] on renaming `let mut` to `var`.

[alt]: https://mail.mozilla.org/pipermail/rust-dev/2014-January/008319.html
