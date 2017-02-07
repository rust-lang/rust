// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepGraph, DepNode, DepTrackingMap, DepTrackingMapConfig};
use hir::def_id::DefId;
use middle::const_val::ConstVal;
use ty::{self, Ty};
use util::nodemap::DefIdSet;

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use syntax::attr;

macro_rules! define_maps {
    ($($(#[$attr:meta])* pub $field:ident: $node_name:ident($key:ty) -> $value:ty),*) => {
        pub struct Maps<'tcx> {
            $($(#[$attr])* pub $field: RefCell<DepTrackingMap<$field<'tcx>>>),*
        }

        impl<'tcx> Maps<'tcx> {
            pub fn new(dep_graph: DepGraph) -> Self {
                Maps {
                    $($field: RefCell::new(DepTrackingMap::new(dep_graph.clone()))),*
                }
            }
        }

        $(#[allow(bad_style)]
        pub struct $field<'tcx> {
            data: PhantomData<&'tcx ()>
        }

        impl<'tcx> DepTrackingMapConfig for $field<'tcx> {
            type Key = $key;
            type Value = $value;
            fn to_dep_node(key: &$key) -> DepNode<DefId> { DepNode::$node_name(*key) }
        })*
    }
}

define_maps! {
    /// Maps from a trait item to the trait item "descriptor"
    pub associated_items: AssociatedItems(DefId) -> ty::AssociatedItem,

    /// Records the type of every item.
    pub types: ItemSignature(DefId) -> Ty<'tcx>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated generics and predicates.
    pub generics: ItemSignature(DefId) -> &'tcx ty::Generics,
    pub predicates: ItemSignature(DefId) -> ty::GenericPredicates<'tcx>,

    /// Maps from the def-id of a trait to the list of
    /// super-predicates. This is a subset of the full list of
    /// predicates. We store these in a separate map because we must
    /// evaluate them even during type conversion, often before the
    /// full predicates are available (note that supertraits have
    /// additional acyclicity requirements).
    pub super_predicates: ItemSignature(DefId) -> ty::GenericPredicates<'tcx>,

    /// Maps from an impl/trait def-id to a list of the def-ids of its items
    pub associated_item_def_ids: AssociatedItemDefIds(DefId) -> Rc<Vec<DefId>>,

    pub impl_trait_refs: ItemSignature(DefId) -> Option<ty::TraitRef<'tcx>>,
    pub trait_defs: ItemSignature(DefId) -> &'tcx ty::TraitDef,
    pub adt_defs: ItemSignature(DefId) -> &'tcx ty::AdtDef,
    pub adt_sized_constraint: SizedConstraint(DefId) -> Ty<'tcx>,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    pub variances: ItemSignature(DefId) -> Rc<Vec<ty::Variance>>,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    pub inherent_impls: InherentImpls(DefId) -> Vec<DefId>,

    /// Caches the representation hints for struct definitions.
    pub repr_hints: ReprHints(DefId) -> Rc<Vec<attr::ReprAttr>>,

    /// Maps from the def-id of a function/method or const/static
    /// to its MIR. Mutation is done at an item granularity to
    /// allow MIR optimization passes to function and still
    /// access cross-crate MIR (e.g. inlining or const eval).
    ///
    /// Note that cross-crate MIR appears to be always borrowed
    /// (in the `RefCell` sense) to prevent accidental mutation.
    pub mir: Mir(DefId) -> &'tcx RefCell<::mir::Mir<'tcx>>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_kinds: ItemSignature(DefId) -> ty::ClosureKind,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_types: ItemSignature(DefId) -> ty::ClosureTy<'tcx>,

    /// Caches CoerceUnsized kinds for impls on custom types.
    pub custom_coerce_unsized_kinds: ItemSignature(DefId)
        -> ty::adjustment::CustomCoerceUnsized,

    pub typeck_tables: TypeckTables(DefId) -> &'tcx ty::TypeckTables<'tcx>,

    /// Set of trait imports actually used in the method resolution.
    /// This is used for warning unused imports.
    pub used_trait_imports: UsedTraitImports(DefId) -> DefIdSet,

    /// Results of evaluating monomorphic constants embedded in
    /// other items, such as enum variant explicit discriminants.
    pub monomorphic_const_eval: MonomorphicConstEval(DefId) -> Result<ConstVal, ()>
}
