// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use util::nodemap::FxHashSet;
use ty::context::TyCtxt;
use ty::{AdtDef, VariantDef, FieldDef, TyS};
use ty::{DefId, Substs};
use ty::{AdtKind, Visibility};
use ty::TypeVariants::*;

pub use self::def_id_forest::DefIdForest;

mod def_id_forest;

// The methods in this module calculate DefIdForests of modules in which a
// AdtDef/VariantDef/FieldDef is visibly uninhabited.
//
// # Example
// ```rust
// enum Void {}
// mod a {
//     pub mod b {
//         pub struct SecretlyUninhabited {
//             _priv: !,
//         }
//     }
// }
//
// mod c {
//     pub struct AlsoSecretlyUninhabited {
//         _priv: Void,
//     }
//     mod d {
//     }
// }
//
// struct Foo {
//     x: a::b::SecretlyUninhabited,
//     y: c::AlsoSecretlyUninhabited,
// }
// ```
// In this code, the type Foo will only be visibly uninhabited inside the
// modules b, c and d. Calling uninhabited_from on Foo or its AdtDef will
// return the forest of modules {b, c->d} (represented in a DefIdForest by the
// set {b, c})
//
// We need this information for pattern-matching on Foo or types that contain
// Foo.
//
// # Example
// ```rust
// let foo_result: Result<T, Foo> = ... ;
// let Ok(t) = foo_result;
// ```
// This code should only compile in modules where the uninhabitedness of Foo is
// visible.

impl<'a, 'gcx, 'tcx> AdtDef {
    /// Calculate the forest of DefIds from which this adt is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>) -> DefIdForest
    {
        if !visited.insert((self.did, substs)) {
            return DefIdForest::empty();
        }

        let ret = DefIdForest::intersection(tcx, self.variants.iter().map(|v| {
            v.uninhabited_from(visited, tcx, substs, self.adt_kind())
        }));
        visited.remove(&(self.did, substs));
        ret
    }
}

impl<'a, 'gcx, 'tcx> VariantDef {
    /// Calculate the forest of DefIds from which this variant is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>,
                adt_kind: AdtKind) -> DefIdForest
    {
        match adt_kind {
            AdtKind::Union => {
                DefIdForest::intersection(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, false)
                }))
            },
            AdtKind::Struct => {
                DefIdForest::union(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, false)
                }))
            },
            AdtKind::Enum => {
                DefIdForest::union(tcx, self.fields.iter().map(|f| {
                    f.uninhabited_from(visited, tcx, substs, true)
                }))
            },
        }
    }
}

impl<'a, 'gcx, 'tcx> FieldDef {
    /// Calculate the forest of DefIds from which this field is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>,
                substs: &'tcx Substs<'tcx>,
                is_enum: bool) -> DefIdForest
    {
        let mut data_uninhabitedness = move || self.ty(tcx, substs).uninhabited_from(visited, tcx);
        // FIXME(canndrew): Currently enum fields are (incorrectly) stored with
        // Visibility::Invisible so we need to override self.vis if we're
        // dealing with an enum.
        if is_enum {
            data_uninhabitedness()
        } else {
            match self.vis {
                Visibility::Invisible => DefIdForest::empty(),
                Visibility::Restricted(from) => {
                    let forest = DefIdForest::from_id(from);
                    let iter = Some(forest).into_iter().chain(Some(data_uninhabitedness()));
                    DefIdForest::intersection(tcx, iter)
                },
                Visibility::Public => data_uninhabitedness(),
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> TyS<'tcx> {
    /// Calculate the forest of DefIds from which this type is visibly uninhabited.
    pub fn uninhabited_from(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>) -> DefIdForest
    {
        match tcx.lift_to_global(&self) {
            Some(global_ty) => {
                {
                    let cache = tcx.inhabitedness_cache.borrow();
                    if let Some(forest) = cache.get(&global_ty) {
                        return forest.clone();
                    }
                }
                let forest = global_ty.uninhabited_from_inner(visited, tcx);
                let mut cache = tcx.inhabitedness_cache.borrow_mut();
                cache.insert(global_ty, forest.clone());
                forest
            },
            None => {
                let forest = self.uninhabited_from_inner(visited, tcx);
                forest
            },
        }
    }

    fn uninhabited_from_inner(
                &self,
                visited: &mut FxHashSet<(DefId, &'tcx Substs<'tcx>)>,
                tcx: TyCtxt<'a, 'gcx, 'tcx>) -> DefIdForest
    {
        match self.sty {
            TyAdt(def, substs) => {
                def.uninhabited_from(visited, tcx, substs)
            },

            TyNever => DefIdForest::full(tcx),
            TyTuple(ref tys) => {
                DefIdForest::union(tcx, tys.iter().map(|ty| {
                    ty.uninhabited_from(visited, tcx)
                }))
            },
            TyArray(ty, len) => {
                if len == 0 {
                    DefIdForest::empty()
                } else {
                    ty.uninhabited_from(visited, tcx)
                }
            }
            TyRef(_, ref tm) => {
                if tcx.sess.features.borrow().never_type {
                    tm.ty.uninhabited_from(visited, tcx)
                } else {
                    DefIdForest::empty()
                }
            }

            _ => DefIdForest::empty(),
        }
    }
}

