pub use self::def_id_forest::DefIdForest;

use crate::ty;
use crate::ty::context::TyCtxt;
use crate::ty::TyKind::*;
use crate::ty::{AdtDef, FieldDef, Ty, VariantDef};
use crate::ty::{AdtKind, Visibility};
use crate::ty::{DefId, SubstsRef};

mod def_id_forest;

// The methods in this module calculate `DefIdForest`s of modules in which an
// `AdtDef`/`VariantDef`/`FieldDef` is visibly uninhabited.
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
// In this code, the type `Foo` will only be visibly uninhabited inside the
// modules `b`, `c` and `d`. Calling `uninhabited_from` on `Foo` or its `AdtDef` will
// return the forest of modules {`b`, `c`->`d`} (represented in a `DefIdForest` by the
// set {`b`, `c`}).
//
// We need this information for pattern-matching on `Foo` or types that contain
// `Foo`.
//
// # Example
// ```rust
// let foo_result: Result<T, Foo> = ... ;
// let Ok(t) = foo_result;
// ```
// This code should only compile in modules where the uninhabitedness of `Foo` is
// visible.

impl<'tcx> TyCtxt<'tcx> {
    /// Checks whether a type is visibly uninhabited from a particular module.
    ///
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
    pub fn is_ty_uninhabited_from(
        self,
        module: DefId,
        ty: Ty<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> bool {
        // To check whether this type is uninhabited at all (not just from the
        // given node), you could check whether the forest is empty.
        // ```
        // forest.is_empty()
        // ```
        ty.uninhabited_from(self, param_env).contains(self, module)
    }
}

impl<'tcx> AdtDef {
    /// Calculates the forest of `DefId`s from which this ADT is visibly uninhabited.
    fn uninhabited_from(
        &self,
        tcx: TyCtxt<'tcx>,
        substs: SubstsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> DefIdForest<'tcx> {
        // Non-exhaustive ADTs from other crates are always considered inhabited.
        if self.is_variant_list_non_exhaustive() && !self.did.is_local() {
            DefIdForest::empty()
        } else {
            DefIdForest::intersection(
                tcx,
                self.variants
                    .iter()
                    .map(|v| v.uninhabited_from(tcx, substs, self.adt_kind(), param_env)),
            )
        }
    }
}

impl<'tcx> VariantDef {
    /// Calculates the forest of `DefId`s from which this variant is visibly uninhabited.
    pub fn uninhabited_from(
        &self,
        tcx: TyCtxt<'tcx>,
        substs: SubstsRef<'tcx>,
        adt_kind: AdtKind,
        param_env: ty::ParamEnv<'tcx>,
    ) -> DefIdForest<'tcx> {
        let is_enum = match adt_kind {
            // For now, `union`s are never considered uninhabited.
            // The precise semantics of inhabitedness with respect to unions is currently undecided.
            AdtKind::Union => return DefIdForest::empty(),
            AdtKind::Enum => true,
            AdtKind::Struct => false,
        };
        // Non-exhaustive variants from other crates are always considered inhabited.
        if self.is_field_list_non_exhaustive() && !self.def_id.is_local() {
            DefIdForest::empty()
        } else {
            DefIdForest::union(
                tcx,
                self.fields.iter().map(|f| f.uninhabited_from(tcx, substs, is_enum, param_env)),
            )
        }
    }
}

impl<'tcx> FieldDef {
    /// Calculates the forest of `DefId`s from which this field is visibly uninhabited.
    fn uninhabited_from(
        &self,
        tcx: TyCtxt<'tcx>,
        substs: SubstsRef<'tcx>,
        is_enum: bool,
        param_env: ty::ParamEnv<'tcx>,
    ) -> DefIdForest<'tcx> {
        let data_uninhabitedness = move || self.ty(tcx, substs).uninhabited_from(tcx, param_env);
        // FIXME(canndrew): Currently enum fields are (incorrectly) stored with
        // `Visibility::Invisible` so we need to override `self.vis` if we're
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
                }
                Visibility::Public => data_uninhabitedness(),
            }
        }
    }
}

impl<'tcx> Ty<'tcx> {
    /// Calculates the forest of `DefId`s from which this type is visibly uninhabited.
    fn uninhabited_from(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> DefIdForest<'tcx> {
        tcx.type_uninhabited_from(param_env.and(self)).clone()
    }
}

// Query provider for `type_uninhabited_from`.
pub(crate) fn type_uninhabited_from<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> DefIdForest<'tcx> {
    let ty = key.value;
    let param_env = key.param_env;
    match *ty.kind() {
        Adt(def, substs) => def.uninhabited_from(tcx, substs, param_env),

        Never => DefIdForest::full(),

        Tuple(ref tys) => DefIdForest::union(
            tcx,
            tys.iter().map(|ty| ty.expect_ty().uninhabited_from(tcx, param_env)),
        ),

        Array(ty, len) => match len.try_eval_usize(tcx, param_env) {
            Some(0) | None => DefIdForest::empty(),
            // If the array is definitely non-empty, it's uninhabited if
            // the type of its elements is uninhabited.
            Some(1..) => ty.uninhabited_from(tcx, param_env),
        },

        // References to uninitialised memory are valid for any type, including
        // uninhabited types, in unsafe code, so we treat all references as
        // inhabited.
        // The precise semantics of inhabitedness with respect to references is currently
        // undecided.
        Ref(..) => DefIdForest::empty(),

        _ => DefIdForest::empty(),
    }
}
