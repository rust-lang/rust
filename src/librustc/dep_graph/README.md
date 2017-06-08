# Dependency graph for incremental compilation

This module contains the infrastructure for managing the incremental
compilation dependency graph. This README aims to explain how it ought
to be used. In this document, we'll first explain the overall
strategy, and then share some tips for handling specific scenarios.

The high-level idea is that we want to instrument the compiler to
track which parts of the AST and other IR are read/written by what.
This way, when we come back later, we can look at this graph and
determine what work needs to be redone.

### The dependency graph

The nodes of the graph are defined by the enum `DepNode`. They represent
one of three things:

1. HIR nodes (like `Hir(DefId)`) represent the HIR input itself.
2. Data nodes (like `ItemSignature(DefId)`) represent some computed
   information about a particular item.
3. Procedure nodes (like `CoherenceCheckTrait(DefId)`) represent some
   procedure that is executing. Usually this procedure is
   performing some kind of check for errors. You can think of them as
   computed values where the value being computed is `()` (and the
   value may fail to be computed, if an error results).

An edge `N1 -> N2` is added between two nodes if either:

- the value of `N1` is used to compute `N2`;
- `N1` is read by the procedure `N2`;
- the procedure `N1` writes the value `N2`.

The latter two conditions are equivalent to the first one if you think
of procedures as values.

### Basic tracking

There is a very general strategy to ensure that you have a correct, if
sometimes overconservative, dependency graph. The two main things you have
to do are (a) identify shared state and (b) identify the current tasks.

### Identifying shared state

Identify "shared state" that will be written by one pass and read by
another. In particular, we need to identify shared state that will be
read "across items" -- that is, anything where changes in one item
could invalidate work done for other items. So, for example:

1. The signature for a function is "shared state".
2. The computed type of some expression in the body of a function is
   not shared state, because if it changes it does not itself
   invalidate other functions (though it may be that it causes new
   monomorphizations to occur, but that's handled independently).

Put another way: if the HIR for an item changes, we are going to
recompile that item for sure. But we need the dep tracking map to tell
us what *else* we have to recompile. Shared state is anything that is
used to communicate results from one item to another.

### Identifying the current task, tracking reads/writes, etc

FIXME(#42293). This text needs to be rewritten for the new red-green
system, which doesn't fully exist yet.

#### Dependency tracking map

`DepTrackingMap` is a particularly convenient way to correctly store
shared state. A `DepTrackingMap` is a special hashmap that will add
edges automatically when `get` and `insert` are called. The idea is
that, when you get/insert a value for the key `K`, we will add an edge
from/to the node `DepNode::Variant(K)` (for some variant specific to
the map).

Each `DepTrackingMap` is parameterized by a special type `M` that
implements `DepTrackingMapConfig`; this trait defines the key and value
types of the map, and also defines a fn for converting from the key to
a `DepNode` label. You don't usually have to muck about with this by
hand, there is a macro for creating it. You can see the complete set
of `DepTrackingMap` definitions in `librustc/middle/ty/maps.rs`.

As an example, let's look at the `adt_defs` map. The `adt_defs` map
maps from the def-id of a struct/enum to its `AdtDef`. It is defined
using this macro:

```rust
dep_map_ty! { AdtDefs: ItemSignature(DefId) -> ty::AdtDefMaster<'tcx> }
//            ~~~~~~~  ~~~~~~~~~~~~~ ~~~~~     ~~~~~~~~~~~~~~~~~~~~~~
//               |           |      Key type       Value type
//               |    DepNode variant
//      Name of map id type
```

this indicates that a map id type `AdtDefs` will be created. The key
of the map will be a `DefId` and value will be
`ty::AdtDefMaster<'tcx>`. The `DepNode` will be created by
`DepNode::ItemSignature(K)` for a given key.

Once that is done, you can just use the `DepTrackingMap` like any
other map:

```rust
let mut map: DepTrackingMap<M> = DepTrackingMap::new(dep_graph);
map.insert(key, value); // registers dep_graph.write
map.get(key; // registers dep_graph.read
```

#### Memoization

One particularly interesting case is memoization. If you have some
shared state that you compute in a memoized fashion, the correct thing
to do is to define a `RefCell<DepTrackingMap>` for it and use the
`memoize` helper:

```rust
map.memoize(key, || /* compute value */)
```

This will create a graph that looks like

    ... -> MapVariant(key) -> CurrentTask

where `MapVariant` is the `DepNode` variant that the map is associated with,
and `...` are whatever edges the `/* compute value */` closure creates.

In particular, using the memoize helper is much better than writing
the obvious code yourself:

```rust
if let Some(result) = map.get(key) {
    return result;
}
let value = /* compute value */;
map.insert(key, value);
```

If you write that code manually, the dependency graph you get will
include artificial edges that are not necessary. For example, imagine that
two tasks, A and B, both invoke the manual memoization code, but A happens
to go first. The resulting graph will be:

    ... -> A -> MapVariant(key) -> B
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~       // caused by A writing to MapVariant(key)
                ~~~~~~~~~~~~~~~~~~~~  // caused by B reading from MapVariant(key)

This graph is not *wrong*, but it encodes a path from A to B that
should not exist.  In contrast, using the memoized helper, you get:

    ... -> MapVariant(key) -> A
                 |
                 +----------> B

which is much cleaner.

**Be aware though that the closure is executed with `MapVariant(key)`
pushed onto the stack as the current task!** That means that you must
add explicit `read` calls for any shared state that it accesses
implicitly from its environment. See the section on "explicit calls to
read and write when starting a new subtask" above for more details.

### How to decide where to introduce a new task

Certainly, you need at least one task on the stack: any attempt to
`read` or `write` shared state will panic if there is no current
task. But where does it make sense to introduce subtasks? The basic
rule is that a subtask makes sense for any discrete unit of work you
may want to skip in the future. Adding a subtask separates out the
reads/writes from *that particular subtask* versus the larger
context. An example: you might have a 'meta' task for all of borrow
checking, and then subtasks for borrow checking individual fns.  (Seen
in this light, memoized computations are just a special case where we
may want to avoid redoing the work even within the context of one
compilation.)

The other case where you might want a subtask is to help with refining
the reads/writes for some later bit of work that needs to be memoized.
For example, we create a subtask for type-checking the body of each
fn.  However, in the initial version of incr. comp. at least, we do
not expect to actually *SKIP* type-checking -- we only expect to skip
trans. However, it's still useful to create subtasks for type-checking
individual items, because, otherwise, if a fn sig changes, we won't
know which callers are affected -- in fact, because the graph would be
so coarse, we'd just have to retrans everything, since we can't
distinguish which fns used which fn sigs.

### Testing the dependency graph

There are various ways to write tests against the dependency graph.
The simplest mechanism are the
`#[rustc_if_this_changed]` and `#[rustc_then_this_would_need]`
annotations. These are used in compile-fail tests to test whether the
expected set of paths exist in the dependency graph. As an example,
see `src/test/compile-fail/dep-graph-caller-callee.rs`.

The idea is that you can annotate a test like:

```rust
#[rustc_if_this_changed]
fn foo() { }

#[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
fn bar() { foo(); }

#[rustc_then_this_would_need(TypeckTables)] //~ ERROR no path
fn baz() { }
```

This will check whether there is a path in the dependency graph from
`Hir(foo)` to `TypeckTables(bar)`. An error is reported for each
`#[rustc_then_this_would_need]` annotation that indicates whether a
path exists. `//~ ERROR` annotations can then be used to test if a
path is found (as demonstrated above).

### Debugging the dependency graph

#### Dumping the graph

The compiler is also capable of dumping the dependency graph for your
debugging pleasure. To do so, pass the `-Z dump-dep-graph` flag. The
graph will be dumped to `dep_graph.{txt,dot}` in the current
directory.  You can override the filename with the `RUST_DEP_GRAPH`
environment variable.

Frequently, though, the full dep graph is quite overwhelming and not
particularly helpful. Therefore, the compiler also allows you to filter
the graph. You can filter in three ways:

1. All edges originating in a particular set of nodes (usually a single node).
2. All edges reaching a particular set of nodes.
3. All edges that lie between given start and end nodes.

To filter, use the `RUST_DEP_GRAPH_FILTER` environment variable, which should
look like one of the following:

```
source_filter     // nodes originating from source_filter
-> target_filter  // nodes that can reach target_filter
source_filter -> target_filter // nodes in between source_filter and target_filter
```

`source_filter` and `target_filter` are a `&`-separated list of strings.
A node is considered to match a filter if all of those strings appear in its
label. So, for example:

```
RUST_DEP_GRAPH_FILTER='-> TypeckTables'
```

would select the predecessors of all `TypeckTables` nodes. Usually though you
want the `TypeckTables` node for some particular fn, so you might write:

```
RUST_DEP_GRAPH_FILTER='-> TypeckTables & bar'
```

This will select only the `TypeckTables` nodes for fns with `bar` in their name.

Perhaps you are finding that when you change `foo` you need to re-type-check `bar`,
but you don't think you should have to. In that case, you might do:

```
RUST_DEP_GRAPH_FILTER='Hir&foo -> TypeckTables & bar'
```

This will dump out all the nodes that lead from `Hir(foo)` to
`TypeckTables(bar)`, from which you can (hopefully) see the source
of the erroneous edge.

#### Tracking down incorrect edges

Sometimes, after you dump the dependency graph, you will find some
path that should not exist, but you will not be quite sure how it came
to be. **When the compiler is built with debug assertions,** it can
help you track that down. Simply set the `RUST_FORBID_DEP_GRAPH_EDGE`
environment variable to a filter. Every edge created in the dep-graph
will be tested against that filter -- if it matches, a `bug!` is
reported, so you can easily see the backtrace (`RUST_BACKTRACE=1`).

The syntax for these filters is the same as described in the previous
section. However, note that this filter is applied to every **edge**
and doesn't handle longer paths in the graph, unlike the previous
section.

Example:

You find that there is a path from the `Hir` of `foo` to the type
check of `bar` and you don't think there should be. You dump the
dep-graph as described in the previous section and open `dep-graph.txt`
to see something like:

    Hir(foo) -> Collect(bar)
    Collect(bar) -> TypeckTables(bar)
    
That first edge looks suspicious to you. So you set
`RUST_FORBID_DEP_GRAPH_EDGE` to `Hir&foo -> Collect&bar`, re-run, and
then observe the backtrace. Voila, bug fixed!
