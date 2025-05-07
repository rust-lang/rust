# Debugging and testing dependencies

## Testing the dependency graph

There are various ways to write tests against the dependency graph.  The
simplest mechanisms are the `#[rustc_if_this_changed]` and
`#[rustc_then_this_would_need]` annotations. These are used in [ui] tests to test
whether the expected set of paths exist in the dependency graph.

[`tests/ui/dep-graph/dep-graph-caller-callee.rs`]: https://github.com/rust-lang/rust/blob/master/tests/ui/dep-graph/dep-graph-caller-callee.rs
[ui]: tests/ui.html

As an example, see [`tests/ui/dep-graph/dep-graph-caller-callee.rs`], or the
tests below.

```rust,ignore
#[rustc_if_this_changed]
fn foo() { }

#[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
fn bar() { foo(); }
```

This should be read as
> If this (`foo`) is changed, then this (i.e. `bar`)'s TypeckTables would need to be changed.

Technically, what occurs is that the test is expected to emit the string "OK" on
stderr, associated to this line.

You could also add the lines

```rust,ignore
#[rustc_then_this_would_need(TypeckTables)] //~ ERROR no path
fn baz() { }
```

Whose meaning is
> If `foo` is changed, then `baz`'s TypeckTables does not need to be changed.
> The macro must emit an error, and the error message must contains "no path".

Recall that the `//~ ERROR OK` is a comment from the point of view of the Rust
code we test, but is meaningful from the point of view of the test itself.

## Debugging the dependency graph

### Dumping the graph

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

```text
source_filter     // nodes originating from source_filter
-> target_filter  // nodes that can reach target_filter
source_filter -> target_filter // nodes in between source_filter and target_filter
```

`source_filter` and `target_filter` are a `&`-separated list of strings.
A node is considered to match a filter if all of those strings appear in its
label. So, for example:

```text
RUST_DEP_GRAPH_FILTER='-> TypeckTables'
```

would select the predecessors of all `TypeckTables` nodes. Usually though you
want the `TypeckTables` node for some particular fn, so you might write:

```text
RUST_DEP_GRAPH_FILTER='-> TypeckTables & bar'
```

This will select only the predecessors of `TypeckTables` nodes for functions
with `bar` in their name.

Perhaps you are finding that when you change `foo` you need to re-type-check
`bar`, but you don't think you should have to. In that case, you might do:

```text
RUST_DEP_GRAPH_FILTER='Hir & foo -> TypeckTables & bar'
```

This will dump out all the nodes that lead from `Hir(foo)` to
`TypeckTables(bar)`, from which you can (hopefully) see the source
of the erroneous edge.

### Tracking down incorrect edges

Sometimes, after you dump the dependency graph, you will find some
path that should not exist, but you will not be quite sure how it came
to be. **When the compiler is built with debug assertions,** it can
help you track that down. Simply set the `RUST_FORBID_DEP_GRAPH_EDGE`
environment variable to a filter. Every edge created in the dep-graph
will be tested against that filter – if it matches, a `bug!` is
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

```text
Hir(foo) -> Collect(bar)
Collect(bar) -> TypeckTables(bar)
```

That first edge looks suspicious to you. So you set
`RUST_FORBID_DEP_GRAPH_EDGE` to `Hir&foo -> Collect&bar`, re-run, and
then observe the backtrace. Voila, bug fixed!
