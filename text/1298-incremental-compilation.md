- Feature Name: incremental-compilation
- Start Date: 2015-08-04
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Enable the compiler to cache incremental workproducts.

# Motivation

The goal of incremental compilation is, naturally, to improve build
times when making small edits. Any reader who has never felt the need
for such a feature is strongly encouraged to attempt hacking on the
compiler or servo sometime (naturally, all readers are so encouraged,
regardless of their opinion on the need for incremental compilation).

## Basic usage

The basic usage will be that one enables incremental compilation using
a compiler flag like `-C incremental-compilation=TMPDIR`. The `TMPDIR`
directory is intended to be an empty directory that the compiler can
use to store intermediate by-products; the compiler will automatically
"GC" this directory, deleting older files that are no longer relevant
and creating new ones.

## High-level design

The high-level idea is that we will track the following intermediate
workproducts for every function (and, indeed, for other kinds of items
as well, but functions are easiest to describe):

- External signature
  - For a function, this would include the types of its arguments,
    where-clauses declared on the function, and so forth.
- MIR
  - The MIR represents the type-checked statements in the body, in
    simplified forms. It is described by [RFC #1211][1211]. As the MIR
    is not fully implemented, this is a non-trivial dependency. We
    could instead use the existing annotated HIR, however that would
    require a larger effort in terms of porting and adapting data
    structures to an incremental setting.  Using the MIR simplifies
    things in this respect.
- Object files
  - This represents the final result of running LLVM. It may be that
    the best strategy is to "cache" compiled code in the form of an
    rlib that is progressively patched, or it may be easier to store
    individual `.o` files that must be relinked (anyone who has worked
    in a substantial C++ project can attest, however, that linking can
    take a non-trivial amount of time).

Of course, the key to any incremental design is to determine what must
be changed. This can be encoded in a *dependency graph*. This graph
connects the various bits of the HIR to the external products
(signatures, MIR, and object files). It is of the utmost importance
that this dependency graph is complete: if edges are missing, the
result will be obscure errors where changes are not fully propagated,
yielding inexplicable behavior at runtime. This RFC proposes an
automatic scheme based on encapsulation.

### Interaction with lints and compiler plugins

Although rustc does not yet support compiler plugins through a stable
interface, we have long planned to allow for custom lints, syntax
extensions, and other sorts of plugins. It would be nice therefore to
be able to accommodate such plugins in the design, so that their
inputs can be tracked and accounted for as well.

## Interaction with optimization

It is important to clarify, though, that this design does not attempt
to enable full optimizing for incremental compilation; indeed the two
are somewhat at odds with one another, as full optimization may
perform inlining and inter-function analysis, which can cause small
edits in one function to affect the generated code of another. This
situation is further exacerbated by the fact that LLVM does not
provide any way to track these sorts of dependencies (e.g., one cannot
even determine what inlining took place, though @dotdash suggested a
clever trick of using llvm lifetime hints). Strategies for handling
this are discussed in the [Optimization section](#optimization) below.

# Detailed design

We begin with a high-level execution plan, followed by sections that
explore aspects of the plan in more detail. The high-level summary
includes links to each of the other sections.

## High-level execution plan

Regardless of whether it is invoked in incremental compilation mode or
not, the compiler will always parse and macro expand the entire crate,
resulting in a HIR tree. Once we have a complete HIR tree, and if we
are invoked in incremental compilation mode, the compiler will then
try to determine which parts of the crate have changed since the last
execution. For each item, we compute a [(mostly) stable id](#defid)
based primarily on the item's name and containing module. We then
compute a hash of its contents and compare that hash against the hash
that the item had in the compilation (if any).

Once we know which items have changed, we consult a
[dependency graph](#depgraph) to tell us which artifacts are still
usable. These artifacts can take the form of serializing MIR graphs,
LLVM IR, compiled object code, and so forth. The dependency graph
tells us which bits of AST contributed to each artifact. It is
constructed by dynamically monitoring what the compiler accesses
during execution.

Finally, we can begin execution. The compiler is currently structured
in a series of passes, each of which walks the entire AST. We do not
need to change this structure to enable incremental
compilation. Instead, we continue to do every pass as normal, but when
we come to an item for which we have a pre-existing artifact (for
example, if we are type-checking a fn that has not changed since the
last execution), we can simply skip over that fn instead. Similar
strategies can be used to enable lazy or parallel compilation at later
times. (Eventually, though, it might be nice to restructure the
compiler so that it operates in more of a demand driven style, rather
than a series of sweeping passes.)

When we come to the final LLVM stages, we must
[separate the functions into distinct "codegen units"](#optimization)
for the purpose of LLVM code generation. This will build on the
existing "codegen-units" used for parallel code generation. LLVM may
perform inlining or interprocedural analysis within a unit, but not
across units, which limits the amount of reoptimization needed when
one of those functions changes.

Finally, the RFC closes with a discussion of
[testing strategies](#testing) we can use to help avoid bugs due to
incremental compilation.

### Staging

One important question is how to stage the incremental compilation
work. That is, it'd be nice to start seeing some benefit as soon as
possible. One possible plan is as follows:

1. Implement stable def-ids (in progress, nearly complete).
2. Implement the dependency graph and tracking system (started).
3. Experiment with distinct modularization schemes to find the one which
   gives the best fragmentation with minimal performance impact.
   Or, at least, implement something finer-grained than today's codegen-units.
4. Persist compiled object code only.
5. Persist intermediate MIR and generated LLVM as well.

The most notable staging point here is that we can begin by just
saving object code, and then gradually add more artifacts that get
saved. The effect of saving fewer things (such as only saving object
code) will simply be to make incremental compilation somewhat less
effective, since we will be forced to re-type-check and re-trans
functions where we might have gotten away with only generating new
object code. However, this is expected to be be a second order effect
overall, particularly since LLVM optimization time can be a very large
portion of compilation.

<a id="defid"></a>
## Handling DefIds

In order to correlate artifacts between compilations, we need some
stable way to name items across compilations (and across crates).  The
compiler currently uses something called a `DefId` to identify each
item. However, these ids today are based on a node-id, which is just
an index into the HIR and hence will change whenever *anything*
preceding it in the HIR changes. We need to make the `DefId` for an
item independent of changes to other items.

Conceptually, the idea is to change `DefId` into the pair of a crate
and a path:

```
DEF_ID = (CRATE, PATH)
CRATE = <crate identifier>
PATH = PATH_ELEM | PATH :: PATH_ELEM
PATH_ELEM = (PATH_ELEM_DATA, <disambiguating integer>)
PATH_ELEM_DATA = Crate(ID)
               | Mod(ID)
               | Item(ID)
               | TypeParameter(ID)
               | LifetimeParameter(ID)
               | Member(ID)
               | Impl
               | ...
```

However, rather than actually store the path in the compiler, we will
instead intern the paths in the `CStore`, and the `DefId` will simply
store an integer. So effectively the `node` field of `DefId`, which
currently indexes into the HIR of the appropriate crate, becomes an
index into the crate's list of paths.

For the most part, these paths match up with user's intuitions. So a
struct `Foo` declared in a module `bar` would just have a path like
`bar::Foo`. However, the paths are also able to express things for
which there is no syntax, such as an item declared within a function
body.

### Disambiguation

For the most part, paths should naturally be unique. However, there
are some cases where a single parent may have multiple children with
the same path. One case would be erroneous programs, where there are
(e.g.) two structs declared with the same name in the same
module. Another is that some items, such as impls, do not have a name,
and hence we cannot easily distinguish them. Finally, it is possible
to declare multiple functions with the same name within function bodies:

```rust
fn foo() {
    {
        fn bar() { }
    }

    {
        fn bar() { }
    }
}
```

All of these cases are handled by a simple *disambiguation* mechanism.
The idea is that we will assign a path to each item as we traverse the
HIR. If we find that a single parent has two children with the same
name, such as two impls, then we simply assign them unique integers in
the order that they appear in the program text. For example, the
following program would use the paths shown (I've elided the
disambiguating integer except where it is relevant):

```rust
mod foo {               // Path: <root>::foo
    pub struct Type { } // Path: <root>::foo::Type
    impl Type {         // Path: <root>::foo::(<impl>,0)
        fn bar() {..}   // Path: <root>::foo::(<impl>,0)::bar
    }
    impl Type { }       // Path: <root>::foo::(<impl>,1)
}
```

Note that the impls were arbitrarily assigned indices based on the order
in which they appear. This does mean that reordering impls may cause
spurious recompilations. We can try to mitigate this somewhat by making the
path entry for an impl include some sort of hash for its header or its contents,
but that will be something we can add later.

*Implementation note:* Refactoring DefIds in this way is a large
task. I've made several attempts at doing it, but my latest branch
appears to be working out (it is not yet complete). As a side benefit,
I've uncovered a few fishy cases where we using the node id from
external crates to index into the local crate's HIR map, which is
certainly incorrect. --nmatsakis
   
<a id="depgraph">
## Identifying and tracking dependencies

### Core idea: a fine-grained dependency graph

Naturally any form of incremental compilation requires a detailed
understanding of how each work item is dependent on other work items.
This is most readily visualized as a dependency graph; the
finer-grained the nodes and edges in this graph, the better. For example,
consider a function `foo` that calls a function `bar`:

```rust
fn foo() {
    ...
    bar();
    ...
}
```

Now imagine that the body (but not the external signature) of `bar`
changes. Do we need to type-check `foo` again? Of course not: `foo`
only cares about the signature of `bar`, not its body. For the
compiler to understand this, though, we'll need to create distinct
graph nodes for the signature and body of each function.

(Note that our policy of making "external signatures" fully explicit
is helpful here. If we supported, e.g., return type inference, than it
would be harder to know whether a change to `bar` means `foo` must be
recompiled.)

### Categories of nodes

This section gives a kind of "first draft" of the set of graph
nodes/edges that we will use. It is expected that the full set of
nodes/edges will evolve in the course of implementation (and of course
over time as well).  In particular, some parts of the graph as
presented here are intentionally quite coarse and we envision that the
graph will be gradually more fine-grained.

The nodes fall into the following categories:

- **HIR nodes.** Represent some portion of the input HIR. For example,
  the body of a fn as a HIR node. These are the inputs to the entire
  compilation process.
  - Examples:
    - `SIG(X)` would represent the signature of some fn item
      `X` that the user wrote (i.e., the names of the types,
      where-clauses, etc)
    - `BODY(X)` would be the body of some fn item `X`
    - and so forth
- **Metadata nodes.** These represent portions of the metadata from
  another crate. Each piece of metadata will include a hash of its
  contents. When we need information about an external item, we load
  that info out of the metadata and add it into the IR nodes below;
  this can be represented in the graph using edges. This means that
  incremental compilation can also work across crates.
- **IR nodes.** Represent some portion of the computed IR. For
  example, the MIR representation of a fn body, or the `ty`
  representation of a fn signature. These also frequently correspond
  to a single entry in one of the various compiler hashmaps. These are
  the outputs (and intermediate steps) of the compilation process
  - Examples:
    - `ITEM_TYPE(X)` -- entry in the obscurely named `tcache` table
      for `X` (what is returned by the rather-more-clearly-named
      `lookup_item_type`)
    - `PREDICATES(X)` -- entry in the `predicates` table
    - `ADT(X)` -- ADT node for a struct (this may want to be more
      fine-grained, particularly to cover the ivars)
    - `MIR(X)` -- the MIR for the item `X`
    - `LLVM(X)` -- the LLVM IR for the item `X`
    - `OBJECT(X)` -- the object code generated by compiling some item
      `X`; the precise way that this is saved will depend on whether
      we use `.o` files that are linked together, or if we attempt to
      amend the shared library in place.
- **Procedure nodes.** These represent various passes performed by the
  compiler. For example, the act of type checking a fn body, or the
  act of constructing MIR for a fn body. These are the "glue" nodes
  that wind up reading the inputs and creating the outputs, and hence
  which ultimately tie the graph together.
  - Examples:
    - `COLLECT(X)` -- the collect code executing on item `X`
    - `WFCHECK(X)` -- the wfcheck code executing on item `X`
    - `BORROWCK(X)` -- the borrowck code executing on item `X`

To see how this all fits together, let's consider the graph for a
simple example:

```rust
fn foo() {
    bar();
}

fn bar() {
}
```

This might generate a graph like the following (the following sections
will describe how this graph is constructed). Note that this is not a
complete graph, it only shows the data needed to produce `MIR(foo)`.

```
BODY(foo) ----------------------------> TYPECK(foo) --> MIR(foo)
                                          ^ ^ ^ ^         |
SIG(foo) ----> COLLECT(foo)               | | | |         |
                 |                        | | | |         v
                 +--> ITEM_TYPE(foo) -----+ | | |      LLVM(foo)
                 +--> PREDICATES(foo) ------+ | |         |
                                              | |         |
SIG(bar) ----> COLLECT(bar)                   | |         v
                 |                            | |     OBJECT(foo)
                 +--> ITEM_TYPE(bar) ---------+ |
                 +--> PREDICATES(bar) ----------+
```

As you can see, this graph indicates that if the signature of either
function changes, we will need to rebuild the MIR for `foo`. But there
is no path from the body of `bar` to the MIR for foo, so changes there
need not trigger a rebuild (we are assuming here that `bar` is not
inlined into `foo`; see the [section on optimizations](#optimization)
for more details on how to handle those sorts of dependencies).

### Building the graph

It is very important the dependency graph contain *all* edges. If any
edges are missing, it will mean that we will get inconsistent builds,
where something should have been rebuilt what was not. Hand-coding a
graph like this, therefore, is probably not the best choice -- we
might get it right at first, but it's easy to for such a setup to fall
out of sync as the code is edited. (For example, if a new table is
added, or a function starts reading data that it didn't before.)

Another consideration is compiler plugins. At present, of course, we
don't have a stable API for such plugins, but eventually we'd like to
support a rich family of them, and they may want to participate in the
incremental compilation system as well. So we need to have an idea of
what data a plugin accesses and modifies, and for what purpose.

The basic strategy then is to build the graph dynamically with an API
that looks something like this:

- `push_procedure(procedure_node)`
- `pop_procedure(procedure_node)`
- `read_from(data_node)`
- `write_to(data_node)`

Here, the `procedure_node` arguments are one of the procedure labels
above (like `COLLECT(X)`), and the `data_node` arguments are either
HIR or IR nodes (e.g., `SIG(X)`, `MIR(X)`).

The idea is that we maintain for each thread a stack of active
procedures. When `push_procedure` is called, a new entry is pushed
onto that stack, and when `pop_procedure` is called, an entry is
popped. When `read_from(D)` is called, we add an edge from `D` to the
top of the stack (it is an error if the stack is empty). Similarly,
`write_to(D)` adds an edge from the top of the stack to `D`.

Naturally it is easy to misuse the above methods: one might forget to
push/pop a procedure at the right time, or fail to invoke
read/write. There are a number of refactorings we can do on the
compiler to make this scheme more robust.

#### Procedures

Most of the compiler passes operate an item at a time. Nonetheless,
they are largely encoded using the standard visitor, which walks all
HIR nodes. We can refactor most of them to instead use an outer
visitor, which walks items, and an inner visitor, which walks a
particular item. (Many passes, such as borrowck, already work this
way.) This outer visitor will be parameterized with the label for the
pass, and will automatically push/pop procedure nodes as appropriate.
This means that as long as you base your pass on the generic
framework, you don't really have to worry.

In general, while I described the general case of a stack of procedure
nodes, it may be desirable to try and maintain the invariant that
there is only ever one procedure node on the stack at a
time. Otherwise, failing to push/pop a procedure at the right time
could result in edges being added to the wrong procedure. It is likely
possible to refactor things to maintain this invariant, but that has
to be determined as we go.

#### IR nodes

Adding edges to the IR nodes that represent the compiler's
intermediate byproducts can be done by leveraging privacy. The idea is
to enforce the use of accessors to the maps and so forth, rather than
allowing direct access. These accessors will call the `read_from` and
`write_to` methods as appropriate to add edges to/from the current
active procedure.

#### HIR nodes

HIR nodes are a bit trickier to encapsulate. After all, the HIR map
itself gives access to the root of the tree, which in turn gives
access to everything else -- and encapsulation is harder to enforce
here.

Some experimentation will be required here, but the rough plan is to:

1. Leveraging the HIR, move away from storing the HIR as one large tree,
   and instead have a tree of items, with each item containing only its own
   content.
   - This way, giving access to the HIR node for an item doesn't implicitly
     give access to all of its subitems.
   - Ideally this would match precisely the HIR nodes we setup, which
     means that e.g. a function would have a subtree corresponding to
     its signature, and a separating subtree corresponding to its
     body.
   - We can still register the lexical nesting of items by linking "indirectly"
     via a `DefId`.
2. Annotate the HIR map accessor methods so that they add appropriate
   read/write edges.

This will integrate with the "default visitor" described under
procedure nodes. This visitor can hand off just an opaque id for each
item, requiring the pass itself to go through the map to fetch the
actual HIR, thus triggering a read edge (we might also bake this
behavior into the visitor for convenience).

### Persisting the graph

Once we've built the graph, we have to persist it, along with some
associated information. The idea is that the compiler, when invoked,
will be supplied with a directory. It will store temporary files in
there. We could also consider extending the design to support use by
multiple simultaneous compiler invocations, which could mean
incremental compilation results even across branches, much like ccache
(but this may require tweaks to the GC strategy).

Once we get to the point of persisting the graph, we don't need the
full details of the graph. The process nodes, in particular, can be
removed. They exist only to create links between the other nodes. To
remove them, we first compute the transitive reachability relationship
and then drop the process nodes out of the graph, leaving only the HIR
nodes (inputs) and IR nodes (output).  (In fact, we only care about
the IR nodes that we intend to persist, which may be only a subset of
the IR nodes, so we can drop those that we do not plan to persist.)

For each HIR node, we will hash the HIR and store that alongside the
node. This indicates precisely the state of the node at the time.
Note that we only need to hash the HIR itself; contextual information
(like `use` statements) that are needed to interpret the text will be
part of a separate HIR node, and there should be edges from that node
to the relevant compiler data structures (such as the name resolution
tables).

For each IR node, we will serialize the relevant information from the
table and store it. The following data will need to be serialized:

- Types, regions, and predicates
- ADT definitions
- MIR definitions
- Identifiers
- Spans

This list was gathered primarily by spelunking through the compiler.
It is probably somewhat incomplete. The appendix below lists an
exhaustive exploration.

### Reusing and garbage collecting artifacts

The general procedure when the compiler starts up in incremental mode
will be to parse and macro expand the input, create the corresponding
set of HIR nodes, and compute their hashes. We can then load the
previous dependency graph and reconcile it against the current state:

- If the dep graph contains a HIR node that is no longer present in the
  source, that node is queued for deletion.
- If the same HIR node exists in both the dep graph and the input, but
  the hash has changed, that node is queued for deletion.
- If there is a HIR node that exists only in the input, it is added
  to the dep graph with no dependencies.

We then delete the transitive closure of nodes queued for deletion
(that is, all the HIR nodes that have changed or been removed, and all
nodes reachable from those HIR nodes). As part of the deletion
process, we remove whatever on disk artifact that may have existed.

<a id="span"></a>
### Handling spans

There are times when the precise span of an item is a significant part
of its metadata. For example, debuginfo needs to identify line numbers
and so forth. However, editing one fn will affect the line numbers for
all subsequent fns in the same file, and it'd be best if we can avoid
recompiling all of them. Our plan is to phase span support in incrementally:

1. Initially, the AST hash will include the filename/line/column,
   which does mean that later fns in the same file will have to be
   recompiled (somewhat unnnecessarily).
2. Eventually, it would be better to encode spans by identifying a
   particular AST node (relative to the root of the item). Since we
   are hashing the structure of the AST, we know the AST from the
   previous and current compilation will match, and thus we can
   compute the current span by finding tha corresponding AST node and
   loading its span. This will require some refactoring and work however.
   
<a id="optimization"></a>
## Optimization and codegen units

There is an inherent tension between incremental compilation and full
optimization. Full optimization may perform inlining and
inter-function analysis, which can cause small edits in one function
to affect the generated code of another. This situation is further
exacerbated by the fact that LLVM does not provide any means to track
when one function was inlined into another, or when some sort of
interprocedural analysis took place (to the best of our knowledge, at
least).

This RFC proposes a simple mechanism for permitting aggressive
optimization, such as inlining, while also supporting reasonable
incremental compilation. The idea is to create *codegen units* that
compartmentalize closely related functions (for example, on a module
boundary). This means that those compartmentalized functions may
analyze one another, while treating functions from other compartments
as opaque entities. This means that when a function in compartment X
changes, we know that functions from other compartments are unaffected
and their object code can be reused. Moreover, while the other
functions in compartment X must be re-optimized, we can still reuse
the existing LLVM IR. (These are the same codegen units as we use for
parallel codegen, but setup differently.)

In terms of the dependency graph, we would create one IR node
representing the codegen unit. This would have the object code as an
associated artifact. We would also have edges from each component of
the codegen unit. As today, generic or inlined functions would not
belong to any codegen unit, but rather would be instantiated anew into
each codegen unit in which they are (transitively) referenced.

There is an analogy here with C++, which naturally faces the same
problems. In that setting, templates and inlineable functions are
often placed into header files. Editing those header files naturally
triggers more recompilation. The compiler could employ a similar
strategy by replicating things that look like good candidates for
inlining into each module; call graphs and profiling information may
be a good input for such heuristics.

<a id="testing"></a>
## Testing strategy

If we are not careful, incremental compilation has the potential to
produce an infinite stream of irreproducible bug reports, so it's
worth considering how we can best test this code.

### Regression tests

The first and most obvious piece of infrastructure is something for
reliable regression testing. The plan is simply to have a series of
sources and patches. The source will have each patch applied in
sequence, rebuilding (incrementally) at each point. We can then check
that (a) we only rebuilt what we expected to rebuild and (b) compare
the result with the result of a fresh build from scratch.  This allows
us to build up tests for specific scenarios or bug reports, but
doesn't help with *finding* bugs in the first place.

### Replaying crates.io versions and git history

The next step is to search across crates.io for consecutive
releases. For a given package, we can checkout version `X.Y` and then
version `X.(Y+1)` and check that incrementally building from one to
the other is successful and that all tests still yield the same
results as before (pass or fail).

A similar search can be performed across git history, where we
identify pairs of consecutive commits. This has the advantage of being
more fine-grained, but the disadvantage of being a MUCH larger search
space.

### Fuzzing

The problem with replaying crates.io versions and even git commits is
that they are probably much larger changes than the typical
recompile. Another option is to use fuzzing, making "innocuous"
changes that should trigger a recompile. Fuzzing is made easier here
because we have an oracle -- that is, we can check that the results of
recompiling incrementally match the results of compiling from scratch.
It's also not necessary that the edits are valid Rust code, though we
should test that too -- in particular, we want to test that the proper
errors are reported when code is invalid, as well.  @nrc also
suggested a clever hybrid, where we use git commits as a source for
the fuzzer's edits, gradually building up the commit.

# Drawbacks

The primary drawback is that incremental compilation may introduce a
new vector for bugs. The design mitigates this concern by attempting
to make the construction of the dependency graph as automated as
possible. We also describe automated testing strategies.

# Alternatives

This design is an evolution from [RFC 594][].

# Unresolved questions

None.

[1211]: https://github.com/rust-lang/rfcs/pull/1211
[RFC 594]: https://github.com/rust-lang/rfcs/pull/594
