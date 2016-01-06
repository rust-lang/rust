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
3. Procedure notes (like `CoherenceCheckImpl(DefId)`) represent some
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

### Identifying the current task

The dep graph always tracks a current task: this is basically the
`DepNode` that the compiler is computing right now. Typically it would
be a procedure node, but it can also be a data node (as noted above,
the two are kind of equivalent).

You set the current task by calling `dep_graph.in_task(node)`. For example:

```rust
let _task = tcx.dep_graph.in_task(DepNode::Privacy);
```

Now all the code until `_task` goes out of scope will be considered
part of the "privacy task".

The tasks are maintained in a stack, so it is perfectly fine to nest
one task within another. Because pushing a task is considered to be
computing a value, when you nest a task `N2` inside of a task `N1`, we
automatically add an edge `N2 -> N1` (since `N1` presumably needed the
result of `N2` to complete):

```rust
let _n1 = tcx.dep_graph.in_task(DepNode::N1);
let _n2 = tcx.dep_graph.in_task(DepNode::N2);
// this will result in an edge N1 -> n2
```

### Ignore tasks

Although it is rarely needed, you can also push a special "ignore"
task:

```rust
let _ignore = tc.dep_graph.in_ignore();
```

This will cause all read/write edges to be ignored until it goes out
of scope or until something else is pushed. For example, we could
suppress the edge between nested tasks like so:

```rust
let _n1 = tcx.dep_graph.in_task(DepNode::N1);
let _ignore = tcx.dep_graph.in_ignore();
let _n2 = tcx.dep_graph.in_task(DepNode::N2);
// now no edge is added
```

### Tracking reads and writes

We need to identify what shared state is read/written by the current
task as it executes. The most fundamental way of doing that is to invoke
the `read` and `write` methods on `DepGraph`:

```rust
// Adds an edge from DepNode::Hir(some_def_id) to the current task
tcx.dep_graph.read(DepNode::Hir(some_def_id))

// Adds an edge from the current task to DepNode::ItemSignature(some_def_id)
tcx.dep_graph.write(DepNode::ItemSignature(some_def_id))
```

However, you should rarely need to invoke those methods directly.
Instead, the idea is to *encapsulate* shared state into some API that
will invoke `read` and `write` automatically. The most common way to
do this is to use a `DepTrackingMap`, described in the next section,
but any sort of abstraction barrier will do. In general, the strategy
is that getting access to information implicitly adds an appropriate
`read`. So, for example, when you use the
`dep_graph::visit_all_items_in_krate` helper method, it will visit
each item `X`, start a task `Foo(X)` for that item, and automatically
add an edge `Hir(X) -> Foo(X)`. This edge is added because the code is
being given access to the HIR node for `X`, and hence it is expected
to read from it. Similarly, reading from the `tcache` map for item `X`
(which is a `DepTrackingMap`, described below) automatically invokes
`dep_graph.read(ItemSignature(X))`.

To make this strategy work, a certain amount of indirection is
required. For example, modules in the HIR do not have direct pointers
to the items that they contain. Rather, they contain node-ids -- one
can then ask the HIR map for the item with a given node-id. This gives
us an opportunity to add an appropriate read edge.

#### Explicit calls to read and write when starting a new subtask

One time when you *may* need to call `read` and `write` directly is
when you push a new task onto the stack, either by calling `in_task`
as shown above or indirectly, such as with the `memoize` pattern
described below. In that case, any data that the task has access to
from the surrounding environment must be explicitly "read". For
example, in `librustc_typeck`, the collection code visits all items
and, among other things, starts a subtask producing its signature
(what follows is simplified pseudocode, of course):

```rust
fn visit_item(item: &hir::Item) {
    // Here, current subtask is "Collect(X)", and an edge Hir(X) -> Collect(X)
    // has automatically been added by `visit_all_items_in_krate`.
    let sig = signature_of_item(item);
}

fn signature_of_item(item: &hir::Item) {
    let def_id = tcx.map.local_def_id(item.id);
    let task = tcx.dep_graph.in_task(DepNode::ItemSignature(def_id));
    tcx.dep_graph.read(DepNode::Hir(def_id)); // <-- the interesting line
    ...
}
```

Here you can see that, in `signature_of_item`, we started a subtask
corresponding to producing the `ItemSignature`. This subtask will read from
`item` -- but it gained access to `item` implicitly. This means that if it just
reads from `item`, there would be missing edges in the graph:

    Hir(X) --+ // added by the explicit call to `read`
      |      |
      |      +---> ItemSignature(X) -> Collect(X)
      |                                 ^
      |                                 |
      +---------------------------------+ // added by `visit_all_items_in_krate`
    
In particular, the edge from `Hir(X)` to `ItemSignature(X)` is only
present because we called `read` ourselves when entering the `ItemSignature(X)`
task.

So, the rule of thumb: when entering a new task yourself, register
reads on any shared state that you inherit. (This actually comes up
fairly infrequently though: the main place you need caution is around
memoization.)

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

```
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

#[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR OK
fn bar() { foo(); }

#[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR no path
fn baz() { }
```

This will check whether there is a path in the dependency graph from
`Hir(foo)` to `TypeckItemBody(bar)`. An error is reported for each
`#[rustc_then_this_would_need]` annotation that indicates whether a
path exists. `//~ ERROR` annotations can then be used to test if a
path is found (as demonstrated above).

### Debugging the dependency graph

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
RUST_DEP_GRAPH_FILTER='-> TypeckItemBody'
```

would select the predecessors of all `TypeckItemBody` nodes. Usually though you
want the `TypeckItemBody` node for some particular fn, so you might write:

```
RUST_DEP_GRAPH_FILTER='-> TypeckItemBody & bar'
```

This will select only the `TypeckItemBody` nodes for fns with `bar` in their name.

Perhaps you are finding that when you change `foo` you need to re-type-check `bar`,
but you don't think you should have to. In that case, you might do:

```
RUST_DEP_GRAPH_FILTER='Hir&foo -> TypeckItemBody & bar'
```

This will dump out all the nodes that lead from `Hir(foo)` to
`TypeckItemBody(bar)`, from which you can (hopefully) see the source
of the erroneous edge.

