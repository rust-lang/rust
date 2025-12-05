# The HIR

The HIR – "High-Level Intermediate Representation" – is the primary IR used
in most of rustc. It is a compiler-friendly representation of the abstract
syntax tree (AST) that is generated after parsing, macro expansion, and name
resolution (see [Lowering](./hir/lowering.md) for how the HIR is created).
Many parts of HIR resemble Rust surface syntax quite closely, with
the exception that some of Rust's expression forms have been desugared away.
For example, `for` loops are converted into a `loop` and do not appear in
the HIR. This makes HIR more amenable to analysis than a normal AST.

This chapter covers the main concepts of the HIR.

You can view the HIR representation of your code by passing the
`-Z unpretty=hir-tree` flag to rustc:

```bash
cargo rustc -- -Z unpretty=hir-tree
```


You can also use the `-Z unpretty=hir` option to generate a HIR
that is closer to the original source code expression:

```bash
cargo rustc -- -Z unpretty=hir
```

## Out-of-band storage and the `Crate` type

The top-level data-structure in the HIR is the [`Crate`], which stores
the contents of the crate currently being compiled (we only ever
construct HIR for the current crate). Whereas in the AST the crate
data structure basically just contains the root module, the HIR
`Crate` structure contains a number of maps and other things that
serve to organize the content of the crate for easier access.

[`Crate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.Crate.html

For example, the contents of individual items (e.g. modules,
functions, traits, impls, etc) in the HIR are not immediately
accessible in the parents. So, for example, if there is a module item
`foo` containing a function `bar()`:

```rust
mod foo {
    fn bar() { }
}
```

then in the HIR the representation of module `foo` (the [`Mod`]
struct) would only have the **`ItemId`** `I` of `bar()`. To get the
details of the function `bar()`, we would lookup `I` in the
`items` map.

[`Mod`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.Mod.html

One nice result from this representation is that one can iterate
over all items in the crate by iterating over the key-value pairs
in these maps (without the need to trawl through the whole HIR).
There are similar maps for things like trait items and impl items,
as well as "bodies" (explained below).

The other reason to set up the representation this way is for better
integration with incremental compilation. This way, if you gain access
to an [`&rustc_hir::Item`] (e.g. for the mod `foo`), you do not immediately
gain access to the contents of the function `bar()`. Instead, you only
gain access to the **id** for `bar()`, and you must invoke some
function to lookup the contents of `bar()` given its id; this gives
the compiler a chance to observe that you accessed the data for
`bar()`, and then record the dependency.

[`&rustc_hir::Item`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.Item.html

<a id="hir-id"></a>

## Identifiers in the HIR

The HIR uses a bunch of different identifiers that coexist and serve different purposes.

- A [`DefId`], as the name suggests, identifies a particular definition, or top-level
  item, in a given crate. It is composed of two parts: a [`CrateNum`] which identifies
  the crate the definition comes from, and a [`DefIndex`] which identifies the definition
  within the crate. Unlike [`HirId`]s, there isn't a [`DefId`] for every expression, which
  makes them more stable across compilations.

- A [`LocalDefId`] is basically a [`DefId`] that is known to come from the current crate.
  This allows us to drop the [`CrateNum`] part, and use the type system to ensure that
  only local definitions are passed to functions that expect a local definition.

- A [`HirId`] uniquely identifies a node in the HIR of the current crate. It is composed
  of two parts: an `owner` and a `local_id` that is unique within the `owner`. This
  combination makes for more stable values which are helpful for incremental compilation.
  Unlike [`DefId`]s, a [`HirId`] can refer to [fine-grained entities][Node] like expressions,
  but stays local to the current crate.

- A [`BodyId`] identifies a HIR [`Body`] in the current crate. It is currently only
  a wrapper around a [`HirId`]. For more info about HIR bodies, please refer to the
  [HIR chapter][hir-bodies].

These identifiers can be converted into one another through the `TyCtxt`.

[`DefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[`LocalDefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.LocalDefId.html
[`HirId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/struct.HirId.html
[`BodyId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.BodyId.html
[Node]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.Node.html
[`CrateNum`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.CrateNum.html
[`DefIndex`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefIndex.html
[`Body`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.Body.html
[hir-bodies]: ./hir.md#hir-bodies

## HIR Operations

Most of the time when you are working with the HIR, you will do so via
`TyCtxt`. It contains a number of methods, defined in the `hir::map` module and
mostly prefixed with `hir_`, to convert between IDs of various kinds and to
lookup data associated with a HIR node.

[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html

For example, if you have a [`LocalDefId`], and you would like to convert it
to a [`HirId`], you can use [`tcx.local_def_id_to_hir_id(def_id)`][local_def_id_to_hir_id].
You need a `LocalDefId`, rather than a `DefId`, since only local items have HIR nodes.

[local_def_id_to_hir_id]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.local_def_id_to_hir_id

Similarly, you can use [`tcx.hir_node(n)`][hir_node] to lookup the node for a
[`HirId`]. This returns a `Option<Node<'hir>>`, where [`Node`] is an enum
defined in the map. By matching on this, you can find out what sort of
node the `HirId` referred to and also get a pointer to the data
itself. Often, you know what sort of node `n` is – e.g. if you know
that `n` must be some HIR expression, you can do
[`tcx.hir_expect_expr(n)`][expect_expr], which will extract and return the
[`&hir::Expr`][Expr], panicking if `n` is not in fact an expression.

[hir_node]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.hir_node
[`Node`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.Node.html
[expect_expr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.expect_expr
[Expr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.Expr.html

Finally, you can find the parents of nodes, via
calls like [`tcx.parent_hir_node(n)`][parent_hir_node].

[parent_hir_node]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.parent_hir_node


## HIR Bodies

A [`rustc_hir::Body`] represents some kind of executable code, such as the body
of a function/closure or the definition of a constant. Bodies are
associated with an **owner**, which is typically some kind of item
(e.g. an `fn()` or `const`), but could also be a closure expression
(e.g. `|x, y| x + y`). You can use the `TyCtxt` to find the body
associated with a given def-id ([`hir_maybe_body_owned_by`]) or to find
the owner of a body ([`hir_body_owner_def_id`]).

[`rustc_hir::Body`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.Body.html
[`hir_maybe_body_owned_by`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.hir_maybe_body_owned_by
[`hir_body_owner_def_id`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.hir_body_owner_def_id
