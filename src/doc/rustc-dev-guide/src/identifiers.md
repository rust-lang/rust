# Identifiers in the compiler

If you have read the few previous chapters, you now know that `rustc` uses
many different intermediate representations to perform different kinds of analyses.
However, like in every data structure, you need a way to traverse the structure
and refer to other elements. In this chapter, you will find information on the
different identifiers `rustc` uses for each intermediate representation.

## In the AST

A [`NodeId`] is an identifier number that uniquely identifies an AST node within
a crate. Every node in the AST has its own [`NodeId`], including top-level items
such as structs, but also individual statements and expressions.

However, because they are absolute within a crate, adding or removing a single
node in the AST causes all the subsequent [`NodeId`]s to change. This renders
[`NodeId`]s pretty much useless for incremental compilation, where you want as
few things as possible to change.

[`NodeId`]s are used in all the `rustc` bits that operate directly on the AST,
like macro expansion and name resolution.

[`NodeId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/node_id/struct.NodeId.html

## In the HIR

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

These identifiers can be converted into one another through the [HIR map][map].
See the [HIR chapter][hir-map] for more detailed information.

[`DefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[`LocalDefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.LocalDefId.html
[`HirId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir_id/struct.HirId.html
[`BodyId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/struct.BodyId.html
[`CrateNum`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.CrateNum.html
[`DefIndex`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefIndex.html
[`Body`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/struct.Body.html
[Node]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.Node.html
[hir-map]: ./hir.md#the-hir-map
[hir-bodies]: ./hir.md#hir-bodies
[map]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/hir/map/struct.Map.html

## In the MIR

- [`BasicBlock`] identifies a *basic block*. It points to an instance of
  [`BasicBlockData`], which can be retrieved by indexing into
  [`Body.basic_blocks`].

- [`Local`] identifies a local variable in a function. Its associated data is in
  [`LocalDecl`], which can be retrieved by indexing into [`Body.local_decls`].

- [`FieldIdx`] identifies a struct's, union's, or enum variant's field. It is used
  as a "projection" in [`Place`].

- [`SourceScope`] identifies a name scope in the original source code. Used for
  diagnostics and for debuginfo in debuggers. It points to an instance of
  [`SourceScopeData`], which can be retrieved by indexing into
  [`Body.source_scopes`].

- [`Promoted`] identifies a promoted constant within another item (related to
  const evaluation). Note: it is unique only locally within the item, so it
  should be associated with a `DefId`.
  [`GlobalId`] will give you a more specific identifier.

- [`GlobalId`] identifies a global variable: a `const`, a `static`, a `const fn`
  where all arguments are [zero-sized types], or a promoted constant.

- [`Location`] represents the location in the MIR of a statement or terminator.
  It identifies the block (using [`BasicBlock`]) and the index of the statement
  or terminator in the block.

[`BasicBlock`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.BasicBlock.html
[`BasicBlockData`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.BasicBlockData.html
[`Body.basic_blocks`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Body.html#structfield.basic_blocks
[`Local`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Local.html
[`LocalDecl`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.LocalDecl.html
[`Body.local_decls`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Body.html#structfield.local_decls
[`FieldIdx`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_abi/struct.FieldIdx.html
[`Place`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Place.html
[`SourceScope`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.SourceScope.html
[`SourceScopeData`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.SourceScopeData.html
[`Body.source_scopes`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Body.html#structfield.source_scopes
[`Promoted`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Promoted.html
[`GlobalId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/struct.GlobalId.html
[`Location`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Location.html
[zero-sized types]: https://doc.rust-lang.org/nomicon/exotic-sizes.html#zero-sized-types-zsts
