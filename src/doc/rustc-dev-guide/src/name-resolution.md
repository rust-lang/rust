# Name resolution

In the previous chapters, we saw how the [*Abstract Syntax Tree* (`AST`)][ast]
is built with all macros expanded. We saw how doing that requires doing some
name resolution to resolve imports and macro names. In this chapter, we show
how this is actually done and more.

[ast]: ./ast-validation.md

In fact, we don't do full name resolution during macro expansion -- we only
resolve imports and macros at that time. This is required to know what to even
expand. Later, after we have the whole AST, we do full name resolution to
resolve all names in the crate. This happens in [`rustc_resolve::late`][late].
Unlike during macro expansion, in this late expansion, we only need to try to
resolve a name once, since no new names can be added. If we fail to resolve a
name, then it is a compiler error.

Name resolution is complex. There are different namespaces (e.g.
macros, values, types, lifetimes), and names may be valid at different (nested)
scopes. Also, different types of names can fail resolution differently, and
failures can happen differently at different scopes. For example, in a module
scope, failure means no unexpanded macros and no unresolved glob imports in
that module. On the other hand, in a function body scope, failure requires that a
name be absent from the block we are in, all outer scopes, and the global
scope.

[late]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/index.html

## Basics

In our programs we refer to variables, types, functions, etc, by giving them
a name. These names are not always unique. For example, take this valid Rust
program:

```rust
type x = u32;
let x: x = 1;
let y: x = 2;
```

How do we know on line 3 whether `x` is a type (`u32`) or a value (1)? These
conflicts are resolved during name resolution. In this specific case, name
resolution defines that type names and variable names live in separate
namespaces and therefore can co-exist.

The name resolution in Rust is a two-phase process. In the first phase, which runs
during `macro` expansion, we build a tree of modules and resolve imports. Macro
expansion and name resolution communicate with each other via the
[`ResolverAstLoweringExt`] trait.

The input to the second phase is the syntax tree, produced by parsing input
files and expanding `macros`. This phase produces links from all the names in the
source to relevant places where the name was introduced. It also generates
helpful error messages, like typo suggestions, traits to import or lints about
unused items.

A successful run of the second phase ([`Resolver::resolve_crate`]) creates kind
of an index the rest of the compilation may use to ask about the present names
(through the `hir::lowering::Resolver` interface).

The name resolution lives in the [`rustc_resolve`] crate, with the bulk in
`lib.rs` and some helpers or symbol-type specific logic in the other modules.

[`Resolver::resolve_crate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Resolver.html#method.resolve_crate
[`ResolverAstLoweringExt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast_lowering/trait.ResolverAstLoweringExt.html
[`rustc_resolve`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/index.html

## Namespaces

Different kind of symbols live in different namespaces ‒ e.g. types don't
clash with variables. This usually doesn't happen, because variables start with
lower-case letter while types with upper-case one, but this is only a
convention. This is legal Rust code that will compile (with warnings):

```rust
type x = u32;
let x: x = 1;
let y: x = 2; // See? x is still a type here.
```

To cope with this, and with slightly different scoping rules for these
namespaces, the resolver keeps them separated and builds separate structures for
them.

In other words, when the code talks about namespaces, it doesn't mean the module
hierarchy, it's types vs. values vs. macros.

## Scopes and ribs

A name is visible only in certain area in the source code. This forms a
hierarchical structure, but not necessarily a simple one ‒ if one scope is
part of another, it doesn't mean a name visible in the outer scope is also
visible in the inner scope, or that it refers to the same thing.

To cope with that, the compiler introduces the concept of [`Rib`]s. This is
an abstraction of a scope. Every time the set of visible names potentially changes,
a new [`Rib`] is pushed onto a stack. The places where this can happen include for
example:

[`Rib`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.Rib.html

* The obvious places ‒ curly braces enclosing a block, function boundaries,
  modules.
* Introducing a `let` binding ‒ this can shadow another binding with the same
  name.
* Macro expansion border ‒ to cope with macro hygiene.

When searching for a name, the stack of [`ribs`] is traversed from the innermost
outwards. This helps to find the closest meaning of the name (the one not
shadowed by anything else). The transition to outer [`Rib`] may also affect
what names are usable ‒ if there are nested functions (not closures),
the inner one can't access parameters and local bindings of the outer one,
even though they should be visible by ordinary scoping rules. An example:

[`ribs`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.LateResolutionVisitor.html#structfield.ribs

```rust
fn do_something<T: Default>(val: T) { // <- New rib in both types and values (1)
    // `val` is accessible, as is the helper function
    // `T` is accessible
   let helper = || { // New rib on the block (2)
        // `val` is accessible here
    }; // End of (2), new rib on `helper` (3)
    // `val` is accessible, `helper` variable shadows `helper` function
    fn helper() { // <- New rib in both types and values (4)
        // `val` is not accessible here, (4) is not transparent for locals
        // `T` is not accessible here
    } // End of (4)
    let val = T::default(); // New rib (5)
    // `val` is the variable, not the parameter here
} // End of (5), (3) and (1)
```

Because the rules for different namespaces are a bit different, each namespace
has its own independent [`Rib`] stack that is constructed in parallel to the others.
In addition, there's also a [`Rib`] stack for local labels (e.g. names of loops or
blocks), which isn't a full namespace in its own right.

## Overall strategy

To perform the name resolution of the whole crate, the syntax tree is traversed
top-down and every encountered name is resolved. This works for most kinds of
names, because at the point of use of a name it is already introduced in the [`Rib`]
hierarchy.

There are some exceptions to this. Items are bit tricky, because they can be
used even before encountered ‒ therefore every block needs to be first scanned
for items to fill in its [`Rib`].

Other, even more problematic ones, are imports which need recursive fixed-point
resolution and macros, that need to be resolved and expanded before the rest of
the code can be processed.

Therefore, the resolution is performed in multiple stages.

## Speculative crate loading

To give useful errors, rustc suggests importing paths into scope if they're
not found. How does it do this? It looks through every module of every crate
and looks for possible matches. This even includes crates that haven't yet
been loaded!

Eagerly loading crates to include import suggestions that haven't yet been
loaded is called _speculative crate loading_, because any errors it encounters
shouldn't be reported: [`rustc_resolve`] decided to load them, not the user. The function
that does this is [`lookup_import_candidates`] and lives in
[`rustc_resolve::diagnostics`].

[`rustc_resolve`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/index.html
[`lookup_import_candidates`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/struct.Resolver.html#method.lookup_import_candidates
[`rustc_resolve::diagnostics`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/diagnostics/index.html

To tell the difference between speculative loads and loads initiated by the
user, [`rustc_resolve`] passes around a `record_used` parameter, which is `false` when
the load is speculative.

## TODO: [#16](https://github.com/rust-lang/rustc-dev-guide/issues/16)

This is a result of the first pass of learning the code. It is definitely
incomplete and not detailed enough. It also might be inaccurate in places.
Still, it probably provides useful first guidepost to what happens in there.

* What exactly does it link to and how is that published and consumed by
  following stages of compilation?
* Who calls it and how it is actually used.
* Is it a pass and then the result is only used, or can it be computed
  incrementally?
* The overall strategy description is a bit vague.
* Where does the name `Rib` come from?
* Does this thing have its own tests, or is it tested only as part of some e2e
  testing?
