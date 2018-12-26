use ty::context::TyCtxt;
use ty::{AdtDef, VariantDef, FieldDef, Ty, TyS};
use ty::{DefId, Substs};
use ty::{AdtKind, Visibility};
use ty::TyKind::*;

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
        ty.uninhabited_from(self)
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
        variant.uninhabited_from(self, substs, adt_kind)
    }
}

impl<'a, 'gcx, 'tcx> AdtDef {
    /// Calculate the forest of DefIds from which this adt is visibly uninhabited.
    fn uninhabited_from(
        &self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        substs: &'tcx Substs<'tcx>) -> DefIdForest
    {
        DefIdForest::intersection(tcx, self.variants.iter().map(|v| {
            v.uninhabited_from(tcx, substs, self.adt_kind())
        }))
    }
}

impl<'a, 'gcx, 'tcx> VariantDef {
    /// Calculate the forest of DefIds from which this variant is visibly uninhabited.
    fn uninhabited_from(
        &self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        substs: &'tcx Substs<'tcx>,
        adt_kind: AdtKind) -> DefIdForest
    {
        let is_enum = match adt_kind {
            // For now, `union`s are never considered uninhabited.
            // The precise semantics of inhabitedness with respect to unions is currently undecided.
            AdtKind::Union => return DefIdForest::empty(),
            AdtKind::Enum => true,
            AdtKind::Struct => false,
        };
        DefIdForest::union(tcx, self.fields.iter().map(|f| {
            f.uninhabited_from(tcx, substs, is_enum)
        }))
    }
}

impl<'a, 'gcx, 'tcx> FieldDef {
    /// Calculate the forest of DefIds from which this field is visibly uninhabited.
    fn uninhabited_from(
        &self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        substs: &'tcx Substs<'tcx>,
        is_enum: bool,
    ) -> DefIdForest {
        let data_uninhabitedness = move || {
            self.ty(tcx, substs).uninhabited_from(tcx)
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
    fn uninhabited_from(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> DefIdForest
    {
        match self.sty {
            Adt(def, substs) => def.uninhabited_from(tcx, substs),

            Never => DefIdForest::full(tcx),

            Tuple(ref tys) => {
                DefIdForest::union(tcx, tys.iter().map(|ty| {
                    ty.uninhabited_from(tcx)
                }))
            }

            Array(ty, len) => {
                match len.assert_usize(tcx) {
                    // If the array is definitely non-empty, it's uninhabited if
                    // the type of its elements is uninhabited.
                    Some(n) if n != 0 => ty.uninhabited_from(tcx),
                    _ => DefIdForest::empty()
                }
            }

            // References to uninitialised memory is valid for any type, including
            // uninhabited types, in unsafe code, so we treat all references as
            // inhabited.
            // The precise semantics of inhabitedness with respect to references is currently
            // undecided.
            Ref(..) => DefIdForest::empty(),

            _ => DefIdForest::empty(),
        }
    }
}

