# Name resolution

The name resolution is a separate pass in the compiler. Its input is the syntax
tree, produced by parsing input files. It produces links from all the names in
the source to relevant places where the name was introduced. It also generates
helpful error messages, like typo suggestions, traits to import or lints about
unused items.

A successful run of the name resolution (`Resolver::resolve_crate`) creates kind
of an index the rest of the compilation may use to ask about the present names
(through the `hir::lowering::Resolver` interface).

The name resolution lives in the `librustc_resolve` crate, with the meat in
`lib.rs` and some helpers or symbol-type specific logic in the other modules.

## Namespaces

Different kind of symbols live in different namespaces ‒ eg. types don't
clash with variables. This usually doesn't happen, because variables start with
lower-case letter while types with upper case one, but this is only a
convention. This is legal Rust code that'll compile (with warnings):

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
part of another, it doesn't mean the name visible in the outer one is also
visible in the inner one, or that it refers to the same thing.

To cope with that, the compiler introduces the concept of Ribs. This is
abstraction of a scope. Every time the set of visible names potentially changes,
a new rib is pushed onto a stack. The places where this can happen includes for
example:

* The obvious places ‒ curly braces enclosing a block, function boundaries,
  modules.
* Introducing a let binding ‒ this can shadow another binding with the same
  name.
* Macro expansion border ‒ to cope with macro hygiene.

When searching for a name, the stack of ribs is traversed from the innermost
outwards. This helps to find the closest meaning of the name (the one not
shadowed by anything else). The transition to outer rib may also change the
rules what names are usable ‒ if there are nested functions (not closures),
the inner one can't access parameters and local bindings of the outer one,
even though they should be visible by ordinary scoping rules. An example:

```rust
fn do_something<T: Default>(val: T) { // <- New rib in both types and values (1)
    // `val` is accessible, as is the helper function
    // `T` is accessible
    let helper = || { // New rib on `helper` (2) and another on the block (3)
        // `val` is accessible here
    }; // End of (3)
    // `val` is accessible, `helper` variable shadows `helper` function
    fn helper() { // <- New rib in both types and values (4)
        // `val` is not accessible here, (4) is not transparent for locals)
        // `T` is not accessible here
    } // End of (4)
    let val = T::default(); // New rib (5)
    // `val` is the variable, not the parameter here
} // End of (5), (2) and (1)
```

Because the rules for different namespaces are a bit different, each namespace
has its own independent rib stack that is constructed in parallel to the others.
In addition, there's also a rib stack for local labels (eg. names of loops or
blocks), which isn't a full namespace in its own right.

## Overall strategy

To perform the name resolution of the whole crate, the syntax tree is traversed
top-down and every encountered name is resolved. This works for most kinds of
names, because at the point of use of a name it is already introduced in the Rib
hierarchy.

There are some exceptions to this. Items are bit tricky, because they can be
used even before encountered ‒ therefore every block needs to be first scanned
for items to fill in its Rib.

Other, even more problematic ones, are imports which need recursive fixed-point
resolution and macros, that need to be resolved and expanded before the rest of
the code can be processed.

Therefore, the resolution is performed in multiple stages.

## TODO:

This is a result of the first pass of learning the code. It is definitely
incomplete and not detailed enough. It also might be inaccurate in places.
Still, it probably provides useful first guidepost to what happens in there.

* What exactly does it link to and how is that published and consumed by
  following stages of compilation?
* Who calls it and how it is actually used.
* Is it a pass and then the result is only used, or can it be computed
  incrementally (eg. for RLS)?
* The overall strategy description is a bit vague.
* Where does the name `Rib` come from?
* Does this thing have its own tests, or is it tested only as part of some e2e
  testing?
