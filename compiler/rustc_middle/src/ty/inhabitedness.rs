use crate::ty::context::TyCtxt;
use crate::ty::{self, DefId, Ty, VariantDef, Visibility};

use rustc_type_ir::sty::TyKind::*;

impl<'tcx> VariantDef {
    fn is_uninhabited_module(
        &self,
        tcx: TyCtxt<'tcx>,
        adt: ty::AdtDef<'_>,
        args: ty::GenericArgsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        module: Option<DefId>,
    ) -> bool {
        debug_assert!(!adt.is_union());
        if self.is_field_list_non_exhaustive() && !self.def_id.is_local() {
            // Non-exhaustive variants from other crates are always considered inhabited.
            return false;
        }
        self.fields.iter().any(|field| {
            let fty = tcx.type_of(field.did).instantiate(tcx, args);
            if adt.is_struct()
                && let Visibility::Restricted(from) = field.vis
                && let Some(module) = module
                && !tcx.is_descendant_of(module, from)
            {
                // The field may be uninhabited, this is not visible from `module`, so we return.
                return false;
            }
            fty.is_uninhabited_module(tcx, param_env, module)
        })
    }

    pub fn is_uninhabited_from(
        &self,
        tcx: TyCtxt<'tcx>,
        adt: ty::AdtDef<'_>,
        args: ty::GenericArgsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        module: DefId,
    ) -> bool {
        self.is_uninhabited_module(tcx, adt, args, param_env, Some(module))
    }

    pub fn is_privately_uninhabited(
        &self,
        tcx: TyCtxt<'tcx>,
        adt: ty::AdtDef<'_>,
        args: ty::GenericArgsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> bool {
        self.is_uninhabited_module(tcx, adt, args, param_env, None)
    }
}

impl<'tcx> Ty<'tcx> {
    pub fn is_trivially_uninhabited(self) -> Option<bool> {
        match self.kind() {
            // For now, unions are always considered inhabited
            Adt(adt, _) if adt.is_union() => Some(false),
            // Non-exhaustive ADTs from other crates are always considered inhabited
            Adt(adt, _) if adt.is_variant_list_non_exhaustive() && !adt.did().is_local() => {
                Some(false)
            }
            Never => Some(true),
            Tuple(tys) if tys.is_empty() => Some(false),
            // use a query for more complex cases
            Param(_) | Alias(..) | Adt(..) | Array(..) | Tuple(_) => None,
            // references and other types are inhabited
            _ => Some(false),
        }
    }

    fn is_uninhabited_module(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        module: Option<DefId>,
    ) -> bool {
        if let Some(trivial) = self.is_trivially_uninhabited() {
            trivial
        } else {
            tcx.type_is_uninhabited_from_raw(param_env.and((self, module)))
        }
    }

    /// Checks whether a type is visibly uninhabited from a particular module.
    ///
    /// # Example
    /// ```
    /// #![feature(never_type)]
    /// # fn main() {}
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
    ///     use super::Void;
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
    /// ```ignore (illustrative)
    /// let foo_result: Result<T, Foo> = ... ;
    /// let Ok(t) = foo_result;
    /// ```
    /// This code should only compile in modules where the uninhabitedness of Foo is
    /// visible.
    #[inline]
    pub fn is_uninhabited_from(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        module: DefId,
    ) -> bool {
        self.is_uninhabited_module(tcx, param_env, Some(module))
    }

    /// Returns true if the type is uninhabited without regard to visibility
    #[inline]
    pub fn is_privately_uninhabited(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> bool {
        self.is_uninhabited_module(tcx, param_env, None)
    }
}
