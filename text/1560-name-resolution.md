- Feature Name: item_like_imports
- Start Date: 2016-02-09
- RFC PR: https://github.com/rust-lang/rfcs/pull/1560
- Rust Issue: https://github.com/rust-lang/rust/issues/35120

# Summary
[summary]: #summary

Some internal and language-level changes to name resolution.

Internally, name resolution will be split into two parts - import resolution and
name lookup. Import resolution is moved forward in time to happen in the same
phase as parsing and macro expansion. Name lookup remains where name resolution
currently takes place (that may change in the future, but is outside the scope
of this RFC). However, name lookup can be done earlier if required (importantly
it can be done during macro expansion to allow using the module system for
macros, also outside the scope of this RFC). Import resolution will use a new
algorithm.

The observable effects of this RFC (i.e., language changes) are some increased
flexibility in the name resolution rules, especially around globs and shadowing.

There is an implementation of the language changes in
[PR #32213](https://github.com/rust-lang/rust/pull/32213).

# Motivation
[motivation]: #motivation

Naming and importing macros currently works very differently to naming and
importing any other item. It would be impossible to use the same rules,
since macro expansion happens before name resolution in the compilation process.
Implementing this RFC means that macro expansion and name resolution can happen
in the same phase, thus allowing macros to use the Rust module system properly.

At the same time, we should be able to accept more Rust programs by tweaking the
current rules around imports and name shadowing. This should make programming
using imports easier.


## Some issues in Rust's name resolution

Whilst name resolution is sometimes considered a simple part of the compiler,
there are some details in Rust which make it tricky to properly specify and
implement. Some of these may seem obvious, but the distinctions will be
important later.

* Imported vs declared names - a name can be imported (e.g., `use foo;`) or
  declared (e.g., `fn foo ...`).
* Single vs glob imports - a name can be explicitly (e.g., `use a::foo;`) or
  implicitly imported (e.g., `use a::*;` where `foo` is declared in `a`).
* Public vs private names - the visibility of names is somewhat tied up with
  name resolution, for example in current Rust `use a::*;` only imports the
  public names from `a`.
* Lexical scoping - a name can be inherited from a surrounding scope, rather
  than being declared in the current one, e.g., `let foo = ...; { foo(); }`.
* There are different kinds of scopes - at the item level, names are not
  inherited from outer modules into inner modules. Items may also be declared
  inside functions and blocks within functions, with different rules from modules.
  At the expression level, blocks (`{...}`) give explicit scope, however, from
  the point of view of macro hygiene and region inference, each `let` statement
  starts a new implicit scope.
* Explicitly declared vs macro generated names - a name can be declared
  explicitly in the source text, or could be declared as the result of expanding
  a macro.
* Rust has multiple namespaces - types, values, and macros exist in separate
  namespaces (some items produce names in multiple namespaces). Imports
  refer (implictly) to one or more names in different namespaces.

  Note that all top-level (i.e., not parameters, etc.) path segments in a path
  other than the last must be in the type namespace, e.g., in `a::b::c`, `a` and
  `b` are assumed to be in the type namespace, and `c` may be in any namespace.
* Rust has an implicit prelude - the prelude defines a set of names which are
  always (unless explicitly opted-out) nameable. The prelude includes macros.
  Names in the prelude can be shadowed by any other names.


# Detailed design
[design]: #detailed-design

## Guiding principles

We would like the following principles to hold. There may be edge cases where
they do not, but we would like these to be as small as possible (and prefer they
don't exist at all).

#### Avoid 'time-travel' ambiguities, or different results of resolution if names
are resolved in different orders.

Due to macro expansion, it is possible for a name to be resolved and then to
become ambiguous, or (with rules formulated in a certain way) for a name to be
resolved, then to be amiguous, then to be resolvable again (possibly to
different bindings).

Furthermore, there is some flexibility in the order in which macros can be
expanded. How a name resolves should be consistent under any ordering.

The strongest form of this principle, I believe, is that at any stage of
macro expansion, and under any ordering of expansions, if a name resolves to a
binding then it should always (i.e., at any other stage of any other expansion
series) resolve to that binding, and if resolving a name produces an error
(n.b., distinct from not being able to resolve), it should always produce an
error.


#### Avoid errors due to the resolver being stuck.

Errors with concrete causes and explanations are easier for the user to
understand and to correct. If an error is caused by name resolution getting
stuck, rather than by a concrete problem, this is hard to explain or correct.

For example, if we support a rule that means that a certain glob can't be
expanded before a macro is, but the macro can only be named via that glob
import, then there is an obvious resolution that can't be reached due to our
ordering constraints.


#### The order of declarations of items should be irrelevant.

I.e., names should be able to be used before they are declared. Note that this
clearly does not hold for declarations of variables in statements inside
function bodies.


#### Macros should be manually expandable.

Compiling a program should have the same result before and after expanding a
macro 'by hand', so long as hygiene is accounted for.


#### Glob imports should be manually expandable.

A programmer should be able to replace a glob import with a list import that
imports any names imported by the glob and used in the current scope, without
changing name resolution behaviour.


#### Visibility should not affect name resolution.

Clearly, visibility affects whether a name can be used or not. However, it
should not affect the mechanics of name resolution. I.e., changing a name from
public to private (or vice versa), should not cause more or fewer name
resolution errors (it may of course cause more or fewer accessibility errors).


## Changes to name resolution rules

### Multiple unused imports

A name may be imported multiple times, it is only a name resolution error if
that name is used. E.g.,

```
mod foo {
    pub struct Qux;
}

mod bar {
    pub struct Qux;
}

mod baz {
    use foo::*;
    use bar::*; // Ok, no name conflict.
}
```

In this example, adding a use of `Qux` in `baz` would cause a name resolution
error.

### Multiple imports of the same binding

A name may be imported multiple times and used if both names bind to the same
item. E.g.,

```
mod foo {
    pub struct Qux;
}

mod bar {
    pub use foo::Qux;
}

mod baz {
    use foo::*;
    use bar::*;

    fn f(q: Qux) {}
}
```

### non-public imports

Currently `use` and `pub use` items are treated differently. Non-public imports
will be treated in the same way as public imports, so they may be referenced
from modules which have access to them. E.g.,

```
mod foo {
    pub struct Qux;
}

mod bar {
    use foo::Qux;

    mod baz {
        use bar::Qux; // Ok
    }
}
```


### Glob imports of accessible but not public names

Glob imports will import all accessible names, not just public ones. E.g.,

```
struct Qux;

mod foo {
    use super::*;

    fn f(q: Qux) {} // Ok
}
```

This change is backwards incompatible. However, the second rule above should
address most cases, e.g.,

```
struct Qux;

mod foo {
    use super::*;
    use super::Qux; // Legal due to the second rule above.

    fn f(q: Qux) {} // Ok
}
```

The below rule (though more controversial) should make this change entirely
backwards compatible.

Note that in combination with the above rule, this means non-public imports are
imported by globs where they are private but accessible.


### Explicit names may shadow implicit names

Here, an implicit name means a name imported via a glob or inherited from an
outer scope (as opposed to being declared or imported directly in an inner scope).

An explicit name may shadow an implicit name without causing a name
resolution error. E.g.,

```
mod foo {
    pub struct Qux;
}

mod bar {
    pub struct Qux;
}

mod baz {
    use foo::*;

    struct Qux; // Shadows foo::Qux.
}

mod boz {
    use foo::*;
    use bar::Qux; // Shadows foo::Qux; note, ordering is not important.
}
```

or

```
fn main() {
    struct Foo; // 1.
    {
        struct Foo; // 2.

        let x = Foo; // Ok and refers to declaration 2.
    }
}
```

Note that shadowing is namespace specific. I believe this is consistent with our
general approach to name spaces. E.g.,

```
mod foo {
    pub struct Qux;
}

mod bar {
    pub trait Qux;
}

mod boz {
    use foo::*;
    use bar::Qux; // Shadows only in the type name space.

    fn f(x: &Qux) {   // bound to bar::Qux.
        let _ = Qux;  // bound to foo::Qux.
    }
}
```

Caveat: an explicit name which is defined by the expansion of a macro does **not**
shadow implicit names. Example:

```
macro_rules! foo {
    () => {
        fn foo() {}
    }
}

mod a {
    fn foo() {}
}

mod b {
    use a::*;

    foo!(); // Expands to `fn foo() {}`, this `foo` does not shadow the `foo`
            // imported from `a` and therefore there is a duplicate name error.
}
```

The rationale for this caveat is so that during import resolution, if we have a
glob import (or other implicit name) we can be sure that any imported names will
not be shadowed, either the name will continue to be valid, or there will be an
error. Without this caveat, a name could be valid, and then after further
expansion, become shadowed by a higher priority name.

An error is reported if there is an ambiguity between names due to the lack of
shadowing, e.g., (this example assumes modularised macros),

```
macro_rules! foo {
    () => {
        macro! bar { ... }
    }
}

mod a {
    macro! bar { ... }
}

mod b {
    use a::*;

    foo!(); // Expands to `macro! bar { ... }`.

    bar!(); // ERROR: bar is ambiguous.
}
```

Note on the caveat: there will only be an error emitted if an ambiguous name is
used directly or indirectly in a macro use. I.e., is the name of a macro that is
used, or is the name of a module that is used to name a macro either in a macro
use or in an import.

Alternatives: we could emit an error even if the ambiguous name is not used, or
as a compromise between these two, we could emit an error if the name is in the
type or macro namespace (a name in the value namespace can never cause problems).

This change is discussed in [issue 31337](https://github.com/rust-lang/rust/issues/31337)
and on this RFC PR's comment thread.


### Re-exports, namespaces, and visibility.

(This is something of a clarification point, rather than explicitly new behaviour.
See also discussion on [issue 31783](https://github.com/rust-lang/rust/issues/31783)).

An import (`use`) or re-export (`pub use`) imports a name in all available
namespaces. E.g., `use a::foo;` will import `foo` in the type and value
namespaces if it is declared in those namespaces in `a`.

For a name to be re-exported, it must be public, e.g, `pub use a::foo;` requires
that `foo` is declared publicly in `a`. This is complicated by namespaces. The
following behaviour should be followed for a re-export of `foo`:

* `foo` is private in all namespaces in which it is declared - emit an error.
* `foo` is public in all namespaces in which it is declared - `foo` is
  re-exported in all namespaces.
* `foo` is mixed public/private - `foo` is re-exported in the namespaces in which
  it is declared publicly and imported but not re-exported in namespaces in which
  it is declared privately.

For a glob re-export, there is an error if there are no public items in any
namespace. Otherwise private names are imported and public names are re-exported
on a per-namespace basis (i.e., following the above rules).

## Changes to the implementation

Note: below I talk about "the binding table", this is sort of hand-waving. I'm
envisaging a sets-of-scopes system where there is effectively a single, global
binding table. However, the details of that are beyond the scope of this RFC.
One can imagine "the binding table" means one binding table per scope, as in the
current system.

Currently, parsing and macro expansion happen in the same phase. With this
proposal, we add import resolution to that mix too. Binding tables as well as
the AST will be produced by libsyntax. Name lookup will continue to be done
where name resolution currently takes place.

To resolve imports, the algorithm proceeds as follows: we start by parsing as
much of the program as we can; like today we don't parse macros. When we find
items which bind a name, we add the name to the binding table. When we find an
import which can't be resolved, we add it to a work list. When we find a glob
import, we have to record a 'back link', so that when a public name is added for
the supplying module, we can add it for the importing module.

We then loop over the work list and try to lookup names. If a name has exactly
one best binding then we use it (and record the binding on a list of resolved
names). If there are zero then we put it back on the work list. If there is more
than one binding, then we record an ambiguity error. When we reach a fixed
point, i.e., the work list no longer changes, then we are done. If the work list
is empty, then expansion/import resolution succeeded, otherwise there are names
not found, or ambiguous names, and we failed.

As we are looking up names, we record the resolutions in the binding table. If
the name we are looking up is for a glob import, we add bindings for every
accessible name currently known.

To expand a macro use, we try to resolve the macro's name. If that fails, we put
it on the work list. Otherwise, we expand that macro by parsing the arguments,
pattern matching, and doing hygienic expansion. We then parse the generated code
in the same way as we parsed the original program. We add new names to the
binding table, and expand any new macro uses.

If we add names for a module which has back links, we must follow them and add
these names to the importing module (if they are accessible).

In pseudo-code:

```
// Assumes parsing is already done, but the two things could be done in the same
// pass.
fn parse_expand_and_resolve() {
    loop until fixed point {
        process_names()
        loop until fixed point {
            process_work_list()
        }
        expand_macros()
    }

    for item in work_list {
        report_error()
    } else {
        success!()
    }
}

fn process_names() {
    // 'module' includes `mod`s, top level of the crate, function bodies
    for each unseen item in any module {
        if item is a definition {
            // struct, trait, type, local variable def, etc.
            bindings.insert(item.name, module, item)
            populate_back_links(module, item)
        } else {
            try_to_resolve_import(module, item)
        }
        record_macro_uses()
    }
}

fn try_to_resolve_import(module, item) {
    if item is an explicit use {
        // item is use a::b::c as d;
        match try_to_resolve(item) {
            Ok(r) => {
                add(bindings.insert(d, module, r, Priority::Explicit))
                populate_back_links(module, item)
            }
            Err() => work_list.push(module, item)
        }
    } else if item is a glob {
        // use a::b::*;
        match try_to_resolve(a::b) {
            Ok(n) => 
                for binding in n {
                    bindings.insert_if_no_higher_priority_binding(binding.name, module, binding, Priority::Glob)
                    populate_back_links(module, binding)
                }
                add_back_link(n to module)
                work_list.remove()
            Err(_) => work_list.push(module, item)
        }
    }    
}

fn process_work_list() {
    for each (module, item) in work_list {
        work_list.remove()
        try_to_resolve_import(module, item)
    }
}
```

Note that this pseudo-code elides some details: that names are imported into
distinct namespaces (the type and value namespaces, and with changes to macro
naming, also the macro namespace), and that we must record whether a name is due
to macro expansion or not to abide by the caveat to the 'explicit names shadow
glob names' rule.

If Rust had a single namespace (or had some other properties), we would not have
to distinguish between failed and unresolved imports. However, it does and we
must. This is not clear from the pseudo-code because it elides namespaces, but
consider the following small example:

```
use a::foo; // foo exists in the value namespace of a.
use b::*;   // foo exists in the type namespace of b.
```

Can we resolve a use of `foo` in type position to the import from `b`? That
depends on whether `foo` exists in the type namespace in `a`. If we can prove
that it does not (i.e., resolution fails) then we can use the glob import. If we
cannot (i.e., the name is unresolved but we can't prove it will not resolve
later), then it is not safe to use the glob import because it may be shadowed by
the explicit import. (Note, since `foo` exists in at least the value namespace
in `a`, there will be no error due to a bad import).

In order to keep macro expansion comprehensible to programmers, we must enforce
that all macro uses resolve to the same binding at the end of resolution as they
do when they were resolved.

We rely on a monotonicity property in macro expansion - once an item exists in a
certain place, it will always exist in that place. It will never disappear and
never change. Note that for the purposes of this property, I do not consider
code annotated with a macro to exist until it has been fully expanded.

A consequence of this is that if the compiler resolves a name, then does some
expansion and resolves it again, the first resolution will still be valid.
However, another resolution may appear, so the resolution of a name may change
as we expand. It can also change from a good resolution to an ambiguity. It is
also possible to change from good to ambiguous to good again. There is even an
edge case where we go from good to ambiguous to the same good resolution (but
via a different route).

If import resolution succeeds, then we check our record of name resolutions. We
re-resolve and check we get the same result. We can also check for un-used
macros at this point.

Note that the rules in the previous section have been carefully formulated to
ensure that this check is sufficient to prevent temporal ambiguities. There are
many slight variations for which this check would not be enough.

### Privacy

In order to resolve imports (and in the future for macro privacy), we must be
able to decide if names are accessible. This requires doing privacy checking as
required during parsing/expansion/import resolution. We can keep the current
algorithm, but check accessibility on demand, rather than as a separate pass.

During macro expansion, once a name is resolvable, then we can safely perform
privacy checking, because parsing and macro expansion will never remove items,
nor change the module structure of an item once it has been expanded.

### Metadata

When a crate is packed into metadata, we must also include the binding table. We
must include private entries due to macros that the crate might export. We don't
need data for function bodies. For functions which are serialised for
inlining/monomorphisation, we should include local data (although it's probably
better to serialise the HIR or MIR, then the local bindings are unnecessary).


# Drawbacks
[drawbacks]: #drawbacks

It's a lot of work and name resolution is complex, therefore there is scope for
introducing bugs.

The macro changes are not backwards compatible, which means having a macro
system 2.0. If users are reluctant to use that, we will have two macro systems
forever.

# Alternatives
[alternatives]: #alternatives

## Naming rules

We could take a subset of the shadowing changes (or none at all), whilst still
changing the implementation of name resolution. In particular, we might want to
discard the explicit/glob shadowing rule change, or only allow items, not
imported names to shadow.

We could also consider different shadowing rules around namespacing. In the
'globs and explicit names' rule change, we could consider an explicit name to
shadow both name spaces and emit a custom error. The example becomes:


```
mod foo {
    pub struct Qux;
}

mod bar {
    pub trait Qux;
}

mod boz {
    use foo::*;
    use bar::Qux; // Shadows both name spaces.

    fn f(x: &Qux) {   // bound to bar::Qux.
        let _ = Qux;  // ERROR, unresolved name Qux; the compiler would emit a
                      // note about shadowing and namespaces.
    }
}
```

## Import resolution algorithm

Rather than lookup names for imports during the fixpoint iteration, one could
save links between imports and definitions. When lookup is required (for macros,
or later in the compiler), these links are followed to find a name, rather than
having the name being immediately available.


# Unresolved questions
[unresolved]: #unresolved-questions

## Name lookup

The name resolution phase would be replaced by a cut-down name lookup phase,
where the binding tables generated during expansion are used to lookup names in
the AST.

We could go further, two appealing possibilities are merging name lookup with
the lowering from AST to HIR, so the HIR is a name-resolved data structure. Or,
name lookup could be done lazily (probably with some caching) so no tables
binding names to definitions are kept. I prefer the first option, but this is
not really in scope for this RFC.

## `pub(restricted)`

Where this RFC touches on the privacy system there are some edge cases involving
the `pub(path)` form of restricted visibility. I expect the precise solutions
will be settled during implementation and this RFC should be amended to reflect
those choices.


# References

* [Niko's prototype](https://github.com/nikomatsakis/rust-name-resolution-algorithm)
* [Blog post](http://ncameron.org/blog/name-resolution/), includes details about
  how the name resolution algorithm interacts with sets of scopes hygiene.
