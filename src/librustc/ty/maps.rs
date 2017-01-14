// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepNode, DepTrackingMapConfig};
use hir::def_id::DefId;
use mir;
use ty::{self, Ty};

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use syntax::attr;

macro_rules! dep_map_ty {
    ($ty_name:ident : $node_name:ident ($key:ty) -> $value:ty) => {
        pub struct $ty_name<'tcx> {
            data: PhantomData<&'tcx ()>
        }

        impl<'tcx> DepTrackingMapConfig for $ty_name<'tcx> {
            type Key = $key;
            type Value = $value;
            fn to_dep_node(key: &$key) -> DepNode<DefId> { DepNode::$node_name(*key) }
        }
    }
}

dep_map_ty! { AssociatedItems: AssociatedItems(DefId) -> ty::AssociatedItem }
dep_map_ty! { Types: ItemSignature(DefId) -> Ty<'tcx> }
dep_map_ty! { Generics: ItemSignature(DefId) -> &'tcx ty::Generics<'tcx> }
dep_map_ty! { Predicates: ItemSignature(DefId) -> ty::GenericPredicates<'tcx> }
dep_map_ty! { SuperPredicates: ItemSignature(DefId) -> ty::GenericPredicates<'tcx> }
dep_map_ty! { AssociatedItemDefIds: AssociatedItemDefIds(DefId) -> Rc<Vec<DefId>> }
dep_map_ty! { ImplTraitRefs: ItemSignature(DefId) -> Option<ty::TraitRef<'tcx>> }
dep_map_ty! { TraitDefs: ItemSignature(DefId) -> &'tcx ty::TraitDef }
dep_map_ty! { AdtDefs: ItemSignature(DefId) -> &'tcx ty::AdtDef }
dep_map_ty! { AdtSizedConstraint: SizedConstraint(DefId) -> Ty<'tcx> }
dep_map_ty! { ItemVariances: ItemSignature(DefId) -> Rc<Vec<ty::Variance>> }
dep_map_ty! { InherentImpls: InherentImpls(DefId) -> Vec<DefId> }
dep_map_ty! { ReprHints: ReprHints(DefId) -> Rc<Vec<attr::ReprAttr>> }
dep_map_ty! { Mir: Mir(DefId) -> &'tcx RefCell<mir::Mir<'tcx>> }
dep_map_ty! { ClosureKinds: ItemSignature(DefId) -> ty::ClosureKind }
dep_map_ty! { ClosureTypes: ItemSignature(DefId) -> ty::ClosureTy<'tcx> }
dep_map_ty! { Tables: Tables(DefId) -> &'tcx ty::Tables<'tcx> }
