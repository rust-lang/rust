- Feature Name: N/A
- Start Date: 2016-02-09
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

# Detailed design
[design]: #detailed-design

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


### Globs and explicit names

An explicit name may shadow a glob imported name without causing a name
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

This change is discussed in [issue 31337](https://github.com/rust-lang/rust/issues/31337).


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
names). If there are zero, or more than one possible binding, then we put it
back on the work list. When we reach a fixed point, i.e., the work list no
longer changes, then we are done. If the work list is empty, then
expansion/import resolution succeeded, otherwise there are names not found, or
ambiguous names, and we failed.

As we are looking up names, we record the resolutions in the binding table. If
the name we are looking up is for a glob import, we add bindings for every
accessible name currently known.

To expand a macro use, we try to resolve the macro's name. If that fails, we put
it on the work list. Otherwise, we expand that macro by parsing the arguments,
pattern matching, and doing hygienic expansion. We then parse the generated code
in the same way as we parsed the original program. We add new names to the
binding table, and expand any new macro uses.

If we add names for a module which has back links, we must follow them and add
these names to the importing module (if they are accessible). When following
these back links, we check for cycles, signaling an error if one is found.

In pseudo-code:

```
// Assumes parsing is already done, but the two things could be done in the same
// pass.
fn parse_expand_and_resolve() {
    loop until fixed point {
        loop until fixed point {
            process_names()
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


# References

* [Niko's prototype](https://github.com/nikomatsakis/rust-name-resolution-algorithm)
* [Blog post](http://ncameron.org/blog/name-resolution/), includes details about
  how the name resolution algorithm interacts with sets of scopes hygiene.
