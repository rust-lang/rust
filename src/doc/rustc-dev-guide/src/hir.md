# The HIR

The HIR – "High-Level Intermediate Representation" – is the primary IR used in
most of rustc. It is a compiler-friendly representation of the abstract syntax
tree (AST) that is generated after parsing, macro expansion, and name
resolution (see [Lowering](./lowering.md) for how the HIR is created).
Many parts of HIR resemble Rust surface syntax quite closely, with
the exception that some of Rust's expression forms have been desugared away. For
example, `for` loops are converted into a `loop` and do not appear in the HIR.
This makes HIR more amenable to analysis than a normal AST.

This chapter covers the main concepts of the HIR.

You can view the HIR representation of your code by passing the
`-Zunpretty=hir-tree` flag to rustc:

```bash
> cargo rustc -- -Zunpretty=hir-tree
```

### Out-of-band storage and the `Crate` type

The top-level data-structure in the HIR is the `Crate`, which stores
the contents of the crate currently being compiled (we only ever
construct HIR for the current crate). Whereas in the AST the crate
data structure basically just contains the root module, the HIR
`Crate` structure contains a number of maps and other things that
serve to organize the content of the crate for easier access.

For example, the contents of individual items (e.g. modules,
functions, traits, impls, etc) in the HIR are not immediately
accessible in the parents. So, for example, if there is a module item
`foo` containing a function `bar()`:

```rust
mod foo {
    fn bar() { }
}
```

then in the HIR the representation of module `foo` (the `Mod`
stuct) would only have the **`ItemId`** `I` of `bar()`. To get the
details of the function `bar()`, we would lookup `I` in the
`items` map.

One nice result from this representation is that one can iterate
over all items in the crate by iterating over the key-value pairs
in these maps (without the need to trawl through the whole HIR).
There are similar maps for things like trait items and impl items,
as well as "bodies" (explained below).

The other reason to set up the representation this way is for better
integration with incremental compilation. This way, if you gain access
to an `&hir::Item` (e.g. for the mod `foo`), you do not immediately
gain access to the contents of the function `bar()`. Instead, you only
gain access to the **id** for `bar()`, and you must invoke some
function to lookup the contents of `bar()` given its id; this gives the
compiler a chance to observe that you accessed the data for `bar()`,
and then record the dependency.

<a name="hir-id"></a>

### Identifiers in the HIR

Most of the code that has to deal with things in HIR tends not to
carry around references into the HIR, but rather to carry around
*identifier numbers* (or just "ids"). Right now, you will find four
sorts of identifiers in active use:

- `DefId`, which primarily names "definitions" or top-level items.
  - You can think of a `DefId` as being shorthand for a very explicit
    and complete path, like `std::collections::HashMap`. However,
    these paths are able to name things that are not nameable in
    normal Rust (e.g. impls), and they also include extra information
    about the crate (such as its version number, as two versions of
    the same crate can co-exist).
  - A `DefId` really consists of two parts, a `CrateNum` (which
    identifies the crate) and a `DefIndex` (which indixes into a list
    of items that is maintained per crate).
- `HirId`, which combines the index of a particular item with an
  offset within that item.
  - the key point of a `HirId` is that it is *relative* to some item
    (which is named via a `DefId`).
- `BodyId`, this is an absolute identifier that refers to a specific
  body (definition of a function or constant) in the crate. It is currently
  effectively a "newtype'd" `NodeId`.
- `NodeId`, which is an absolute id that identifies a single node in the HIR
  tree.
  - While these are still in common use, **they are being slowly phased out**.
  - Since they are absolute within the crate, adding a new node anywhere in the
    tree causes the `NodeId`s of all subsequent code in the crate to change.
    This is terrible for incremental compilation, as you can perhaps imagine.

### The HIR Map

Most of the time when you are working with the HIR, you will do so via
the **HIR Map**, accessible in the tcx via `tcx.hir` (and defined in
the `hir::map` module). The HIR map contains a number of methods to
convert between IDs of various kinds and to lookup data associated
with an HIR node.

For example, if you have a `DefId`, and you would like to convert it
to a `NodeId`, you can use `tcx.hir.as_local_node_id(def_id)`. This
returns an `Option<NodeId>` – this will be `None` if the def-id
refers to something outside of the current crate (since then it has no
HIR node), but otherwise returns `Some(n)` where `n` is the node-id of
the definition.

Similarly, you can use `tcx.hir.find(n)` to lookup the node for a
`NodeId`. This returns a `Option<Node<'tcx>>`, where `Node` is an enum
defined in the map; by matching on this you can find out what sort of
node the node-id referred to and also get a pointer to the data
itself. Often, you know what sort of node `n` is – e.g. if you know
that `n` must be some HIR expression, you can do
`tcx.hir.expect_expr(n)`, which will extract and return the
`&hir::Expr`, panicking if `n` is not in fact an expression.

Finally, you can use the HIR map to find the parents of nodes, via
calls like `tcx.hir.get_parent_node(n)`.

### HIR Bodies

A **body** represents some kind of executable code, such as the body
of a function/closure or the definition of a constant. Bodies are
associated with an **owner**, which is typically some kind of item
(e.g. an `fn()` or `const`), but could also be a closure expression
(e.g. `|x, y| x + y`). You can use the HIR map to find the body
associated with a given def-id (`maybe_body_owned_by()`) or to find
the owner of a body (`body_owner_def_id()`).
