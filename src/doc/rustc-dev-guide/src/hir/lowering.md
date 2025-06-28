# AST lowering

The AST lowering step converts AST to [HIR](../hir.md).
This means many structures are removed if they are irrelevant
for type analysis or similar syntax agnostic analyses. Examples
of such structures include but are not limited to

* Parenthesis
    * Removed without replacement, the tree structure makes order explicit
* `for` loops and `while (let)` loops
    * Converted to `loop` + `match` and some `let` bindings
* `if let`
    * Converted to `match`
* Universal `impl Trait`
    * Converted to generic arguments
      (but with some flags, to know that the user didn't write them)
* Existential `impl Trait`
    * Converted to a virtual `existential type` declaration

Lowering needs to uphold several invariants in order to not trigger the
sanity checks in `compiler/rustc_passes/src/hir_id_validator.rs`:

1. A `HirId` must be used if created. So if you use the `lower_node_id`,
  you *must* use the resulting `NodeId` or `HirId` (either is fine, since
  any `NodeId`s in the `HIR` are checked for existing `HirId`s)
2. Lowering a `HirId` must be done in the scope of the *owning* item.
  This means you need to use `with_hir_id_owner` if you are creating parts
  of an item other than the one being currently lowered. This happens for
  example during the lowering of existential `impl Trait`
3. A `NodeId` that will be placed into a HIR structure must be lowered,
  even if its `HirId` is unused. Calling
  `let _ = self.lower_node_id(node_id);` is perfectly legitimate.
4. If you are creating new nodes that didn't exist in the `AST`, you *must*
  create new ids for them. This is done by calling the `next_id` method,
  which produces both a new `NodeId` as well as automatically lowering it
  for you so you also get the `HirId`.

If you are creating new `DefId`s, since each `DefId` needs to have a
corresponding `NodeId`, it is advisable to add these `NodeId`s to the
`AST` so you don't have to generate new ones during lowering. This has
the advantage of creating a way to find the `DefId` of something via its
`NodeId`. If lowering needs this `DefId` in multiple places, you can't
generate a new `NodeId` in all those places because you'd also get a new
`DefId` then. With a `NodeId` from the `AST` this is not an issue.

Having the `NodeId` also allows the `DefCollector` to generate the `DefId`s
instead of lowering having to do it on the fly. Centralizing the `DefId`
generation in one place makes it easier to refactor and reason about.
