// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use util::nodemap::{FxHashMap, FxHashSet};
use ty::context::TyCtxt;
use ty::{AdtDef, VariantDef, FieldDef, Ty, TyS};
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

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Checks whether a type is visibly uninhabited from a particular module.
    /// # Example
    /// ```rust
    /// enum Void {}
    /// mod a {
    ///     pub mod b {
    ///         pub struct SecretlyUninhabited {
    ///             _priv: !,
    ///         }
    ///     }
    /// }
    ///
    /// mod c {
    ///     pub struct AlsoSecretlyUninhabited {
    ///         _priv: Void,
    ///     }
    ///     mod d {
    ///     }
    /// }
    ///
    /// struct Foo {
    ///     x: a::b::SecretlyUninhabited,
    ///     y: c::AlsoSecretlyUninhabited,
    /// }
    /// ```
    /// In this code, the type `Foo` will only be visibly uninhabited inside the
    /// modules b, c and d. This effects pattern-matching on `Foo` or types that
    /// contain `Foo`.
    ///
    /// # Example
    /// ```rust
    /// let foo_result: Result<T, Foo> = ... ;
    /// let Ok(t) = foo_result;
    /// ```
    /// This code should only compile in modules where the uninhabitedness of Foo is
    /// visible.
    pub fn is_ty_uninhabited_from(self, module: DefId, ty: Ty<'tcx>) -> bool {
        // To check whether this type is uninhabited at all (not just from the
        // given node) you could check whether the forest is empty.
        // ```
        // forest.is_empty()
        // ```
        self.ty_inhabitedness_forest(ty).contains(self, module)
    }

    pub fn is_ty_uninhabited_from_all_modules(self, ty: Ty<'tcx>) -> bool {
        !self.ty_inhabitedness_forest(ty).is_empty()
    }

    fn ty_inhabitedness_forest(self, ty: Ty<'tcx>) -> DefIdForest {
        ty.uninhabited_from(&mut FxHashMap(), self)
    }

    pub fn is_enum_variant_uninhabited_from(self,
                                            module: DefId,
                                            variant: &'tcx VariantDef,
                                            substs: &'tcx Substs<'tcx>)
                                            -> bool
    {
        self.variant_inhabitedness_forest(variant, substs).contains(self, module)
    }

    pub fn is_variant_uninhabited_from_all_modules(self,
                                                   variant: &'tcx VariantDef,
                                                   substs: &'tcx Substs<'tcx>)
                                                   -> bool
    {
        !self.variant_inhabitedness_forest(variant, substs).is_empty()
    }

    fn variant_inhabitedness_forest(self, variant: &'tcx VariantDef, substs: &'tcx Substs<'tcx>)
                                    -> DefIdForest {
        // Determine the ADT kind:
        let adt_def_id = self.adt_def_id_of_variant(variant);
        let adt_kind = self.adt_def(adt_def_id).adt_kind();

        // Compute inhabitedness forest:
        variant.uninhabited_from(&mut FxHashMap(), self, substs, adt_kind)
    }
}

impl<'a, 'gcx, 'tcx> AdtDef {
    /// Calculate the forest of DefIds from which this adt is visibly uninhabited.
    fn uninhabited_from(
        &self,
        visited: &mut FxHashMap<DefId, FxHashSet<&'tcx Substs<'tcx>>>,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        substs: &'tcx Substs<'tcx>) -> DefIdForest
    {
        DefIdForest::intersection(tcx, self.variants.iter().map(|v| {
            v.uninhabited_from(visited, tcx, substs, self.adt_kind())
        }))
    }
}

impl<'a, 'gcx, 'tcx> VariantDef {
    /// Calculate the forest of DefIds from which this variant is visibly uninhabited.
    fn uninhabited_from(
        &self,
        visited: &mut FxHashMap<DefId, FxHashSet<&'tcx Substs<'tcx>>>,
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
    fn uninhabited_from(
        &self,
        visited: &mut FxHashMap<DefId, FxHashSet<&'tcx Substs<'tcx>>>,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        substs: &'tcx Substs<'tcx>,
        is_enum: bool) -> DefIdForest
    {
        let mut data_uninhabitedness = move || {
            self.ty(tcx, substs).uninhabited_from(visited, tcx)
        };
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
    fn uninhabited_from(
        &self,
        visited: &mut FxHashMap<DefId, FxHashSet<&'tcx Substs<'tcx>>>,
        tcx: TyCtxt<'a, 'gcx, 'tcx>) -> DefIdForest
    {
        match self.sty {
            TyAdt(def, substs) => {
                {
                    let substs_set = visited.entry(def.did).or_insert(FxHashSet::default());
                    if !substs_set.insert(substs) {
                        // We are already calculating the inhabitedness of this type.
                        // The type must contain a reference to itself. Break the
                        // infinite loop.
                        return DefIdForest::empty();
                    }
                    if substs_set.len() >= tcx.sess.recursion_limit.get() / 4 {
                        // We have gone very deep, reinstantiating this ADT inside
                        // itself with different type arguments. We are probably
                        // hitting an infinite loop. For example, it's possible to write:
                        //                a type Foo<T>
                        //      which contains a Foo<(T, T)>
                        //      which contains a Foo<((T, T), (T, T))>
                        //      which contains a Foo<(((T, T), (T, T)), ((T, T), (T, T)))>
                        //      etc.
                        let error = format!("reached recursion limit while checking \
                                             inhabitedness of `{}`", self);
                        tcx.sess.fatal(&error);
                    }
                }
                let ret = def.uninhabited_from(visited, tcx, substs);
                let substs_set = visited.get_mut(&def.did).unwrap();
                substs_set.remove(substs);
                ret
            },

            TyNever => DefIdForest::full(tcx),
            TyTuple(ref tys, _) => {
                DefIdForest::union(tcx, tys.iter().map(|ty| {
                    ty.uninhabited_from(visited, tcx)
                }))
            },
            TyArray(ty, len) => {
                match len.val.to_u128() {
                    // If the array is definitely non-empty, it's uninhabited if
                    // the type of its elements is uninhabited.
                    Some(n) if n != 0 => ty.uninhabited_from(visited, tcx),
                    _ => DefIdForest::empty()
                }
            }
            TyRef(_, ref tm) => {
                tm.ty.uninhabited_from(visited, tcx)
            }

            _ => DefIdForest::empty(),
        }
    }
}

