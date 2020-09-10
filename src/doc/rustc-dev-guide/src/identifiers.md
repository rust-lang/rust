# Identifiers in the Compiler

If you have read the few previous chapters, you now know that the `rustc` uses
many different intermediate representations to perform different kinds of analysis.
However, like in every data structure, you need a way to traverse the structure
and refer to other elements. In this chapter, you will find information on the
different identifiers `rustc` uses for each intermediate representation.

## In the AST

A [`NodeId`] is an identifier number that uniquely identifies an AST node within
a crate. Every node in the AST has its own [`NodeId`], including top-level items
such as structs, but also individual statements and expressions.

However, because they are absolute within in a crate, adding or removing a single
node in the AST causes all the subsequent [`NodeId`]s to change. This renders
[`NodeId`]s pretty much useless for incremental compilation, where you want as
few things as possible to change.

[`NodeId`]s are used in all the `rustc` bits that operate directly on the AST,
like macro expansion and name resolution.

[`NodeId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/node_id/struct.NodeId.html

## In the HIR

The HIR uses a bunch of different identifiers that coexist and serve different purposes.

- A [`DefId`], as the name suggests, identifies a particular definition, or top-level
  item, in a given grate. It is composed of two parts: a [`CrateNum`] which identifies
  the crate the definition comes from, and a [`DefIndex`] which identifies the definition
  within the crate. Unlike [`NodeId`]s, there isn't a [`DefId`] for every expression, which
  makes them more stable across compilations.
- A [`LocalDefId`] is basically a [`DefId`] that is known to come from the current crate.
  This allows us to drop the [`CrateNum`] part, and use the type system to ensure that
  only local definitions are passed to functions that expect a local definition.
- A [`HirId`] uniquely identifies a node in the HIR of the current crate. It is composed
  of two parts: an `owner` and a `local_id` that is unique within the `owner`. This
  combination makes for more stable values which are helpful for incremental compilation.
  Unlike [`DefId`]s, a [`HirId`] can refer to [fine-grained entities][Node] like expressions,
  but stays local to the current crate.
- A [`BodyId`] identifies a HIR [`Body`] in the current crate. It is currenty only
  a wrapper around a [`HirId`]. For more info about HIR bodies, please refer to the
  [HIR chapter][hir-bodies].

These identifiers can be converted into one another through the [HIR map][map].
See the [HIR chapter][hir-map] for more detailed information.

[`DefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[`LocalDefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.LocalDefId.html
[`HirId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir_id/struct.HirId.html
[`BodyId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/struct.BodyId.html
[`CrateNum`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/enum.CrateNum.html
[`DefIndex`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefIndex.html
[`Body`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/struct.Body.html
[Node]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.Node.html
[hir-map]: ./hir.md#the-hir-map
[hir-bodies]: ./hir.md#hir-bodies
[map]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/hir/map/struct.Map.html

## In the MIR

**TODO**
