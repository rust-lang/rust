# AST lowering

The AST lowering step converts AST to [HIR](../hir.md).
This means many structures are removed if they are irrelevant
for type analysis or similar syntax-agnostic analyses.
Examples of such structures include but are not limited to

* Parenthesis
    * Removed without replacement, the tree structure makes order explicit
* `for` loops
    * Converted to `match` + `loop` + `match`
* Universal `impl Trait`
    * Converted to generic arguments
      (but with some flags, to know that the user didn't write them)
* Existential `impl Trait`
    * Converted to a virtual `existential type` declaration

The implementation of AST lowering is in the [`rustc_ast_lowering`] crate.
The entry point is [`lower_to_hir`], which retrieves the post-expansion AST
and resolver data from [`TyCtxt`] and builds the [`hir::Crate`] for the whole crate.

Lowering is organized around HIR owners.
[`lower_to_hir`] first indexes the
crate and then [`ItemLowerer::lower_node`] lowers each crate, item, associated
item, and foreign item.

Most of the lowering logic lives on [`LoweringContext`].
The implementation is
split across multiple files in the [`rustc_ast_lowering`] crate such as `item.rs`,
`expr.rs`, `pat.rs`, `path.rs`, and others, but they all share the same [`LoweringContext`]
state and ID‑lowering machinery.

Each owner is lowered in its own [`LoweringContext`] context.
Each Lowering owner gets a new lowering context inside `lower_to_hir`,
specifically `with_lctx` helper function.
This is why the `HirId` invariants below matter: `lower_node_id` allocates a new `HirId` for
the given `NodeId`, while `next_id` creates fresh `HirId` introduced during desugaring.

Lowering needs to uphold several invariants in order to not trigger the
sanity checks in [`compiler/rustc_passes/src/hir_id_validator.rs`][hir_id_validator]:

1. A `HirId` must be used if created,
   therefore the `ItemLocalId`s set of local IDs associated with the owner’s nodes
   form a dense range of integers starting at zero as IDs are allocated
   consecutively starting at one, zero is reserved for the owner.

2. Lowering a `HirId` must be done in the scope of the *owning* item.
   And this is done by creating a new `LoweringContext` for each owner.

3. When lowering an AST node,
   the compiler can create HIR nodes that don't exist in the AST tree during desugaring
   such as the loop instead of the for loop.
   These HIR nodes must have a `HirId` and this is done by `next_id`.

4. There's always at least one entry for the owning item itself.
   The owner node has local ID zero, so the owner’s collection is nonempty.

5. When traversing an item with a nested item,
   the owner part of the nested item's definition parent `HirId` must match
   the enclosing HIR owner currently being traversed.

[`rustc_ast_lowering`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast_lowering/index.html
[`lower_to_hir`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast_lowering/fn.lower_to_hir.html
[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html
[`hir::Crate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/hir/struct.Crate.html
[`ItemLowerer::lower_node`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast_lowering/item/struct.ItemLowerer.html
[`LoweringContext`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast_lowering/struct.LoweringContext.html
[hir_id_validator]: https://github.com/rust-lang/rust/blob/HEAD/compiler/rustc_passes/src/hir_id_validator.rs

If you are creating new `DefId`s, since each `DefId` needs to have a
corresponding `NodeId`, it is advisable to add these `NodeId`s to the
`AST` so you don't have to generate new ones during lowering.
This has
the advantage of creating a way to find the `DefId` of something via its
`NodeId`.
If lowering needs this `DefId` in multiple places, you can't
generate a new `NodeId` in all those places because you'd also get a new
`DefId` then.
With a `NodeId` from the `AST`, this is not an issue.

Having the `NodeId` also allows the `DefCollector` to generate the `DefId`s
instead of lowering having to do it on the fly.
Centralizing the `DefId`
generation in one place makes it easier to refactor and reason about.
