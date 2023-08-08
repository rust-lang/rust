//! Implementation of Chalk debug helper functions using TLS.
use std::fmt::{self, Display};

use itertools::Itertools;

use crate::{
    chalk_db, db::HirDatabase, from_assoc_type_id, from_chalk_trait_id, mapping::from_chalk,
    CallableDefId, Interner, ProjectionTyExt,
};
use hir_def::{AdtId, ItemContainerId, Lookup, TypeAliasId};

pub(crate) use unsafe_tls::{set_current_program, with_current_program};

pub(crate) struct DebugContext<'a>(&'a dyn HirDatabase);

impl DebugContext<'_> {
    pub(crate) fn debug_struct_id(
        &self,
        id: chalk_db::AdtId,
        f: &mut fmt::Formatter<'_>,
    ) -> Result<(), fmt::Error> {
        let name = match id.0 {
            AdtId::StructId(it) => self.0.struct_data(it).name.clone(),
            AdtId::UnionId(it) => self.0.union_data(it).name.clone(),
            AdtId::EnumId(it) => self.0.enum_data(it).name.clone(),
        };
        name.display(self.0.upcast()).fmt(f)?;
        Ok(())
    }

    pub(crate) fn debug_trait_id(
        &self,
        id: chalk_db::TraitId,
        f: &mut fmt::Formatter<'_>,
    ) -> Result<(), fmt::Error> {
        let trait_: hir_def::TraitId = from_chalk_trait_id(id);
        let trait_data = self.0.trait_data(trait_);
        trait_data.name.display(self.0.upcast()).fmt(f)?;
        Ok(())
    }

    pub(crate) fn debug_assoc_type_id(
        &self,
        id: chalk_db::AssocTypeId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Result<(), fmt::Error> {
        let type_alias: TypeAliasId = from_assoc_type_id(id);
        let type_alias_data = self.0.type_alias_data(type_alias);
        let trait_ = match type_alias.lookup(self.0.upcast()).container {
            ItemContainerId::TraitId(t) => t,
            _ => panic!("associated type not in trait"),
        };
        let trait_data = self.0.trait_data(trait_);
        write!(
            fmt,
            "{}::{}",
            trait_data.name.display(self.0.upcast()),
            type_alias_data.name.display(self.0.upcast())
        )?;
        Ok(())
    }

    pub(crate) fn debug_projection_ty(
        &self,
        projection_ty: &chalk_ir::ProjectionTy<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Result<(), fmt::Error> {
        let type_alias = from_assoc_type_id(projection_ty.associated_ty_id);
        let type_alias_data = self.0.type_alias_data(type_alias);
        let trait_ = match type_alias.lookup(self.0.upcast()).container {
            ItemContainerId::TraitId(t) => t,
            _ => panic!("associated type not in trait"),
        };
        let trait_name = &self.0.trait_data(trait_).name;
        let trait_ref = projection_ty.trait_ref(self.0);
        let trait_params = trait_ref.substitution.as_slice(Interner);
        let self_ty = trait_ref.self_type_parameter(Interner);
        write!(fmt, "<{self_ty:?} as {}", trait_name.display(self.0.upcast()))?;
        if trait_params.len() > 1 {
            write!(
                fmt,
                "<{}>",
                trait_params[1..].iter().format_with(", ", |x, f| f(&format_args!("{x:?}"))),
            )?;
        }
        write!(fmt, ">::{}", type_alias_data.name.display(self.0.upcast()))?;

        let proj_params_count = projection_ty.substitution.len(Interner) - trait_params.len();
        let proj_params = &projection_ty.substitution.as_slice(Interner)[..proj_params_count];
        if !proj_params.is_empty() {
            write!(
                fmt,
                "<{}>",
                proj_params.iter().format_with(", ", |x, f| f(&format_args!("{x:?}"))),
            )?;
        }

        Ok(())
    }

    pub(crate) fn debug_fn_def_id(
        &self,
        fn_def_id: chalk_ir::FnDefId<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Result<(), fmt::Error> {
        let def: CallableDefId = from_chalk(self.0, fn_def_id);
        let name = match def {
            CallableDefId::FunctionId(ff) => self.0.function_data(ff).name.clone(),
            CallableDefId::StructId(s) => self.0.struct_data(s).name.clone(),
            CallableDefId::EnumVariantId(e) => {
                let enum_data = self.0.enum_data(e.parent);
                enum_data.variants[e.local_id].name.clone()
            }
        };
        match def {
            CallableDefId::FunctionId(_) => write!(fmt, "{{fn {}}}", name.display(self.0.upcast())),
            CallableDefId::StructId(_) | CallableDefId::EnumVariantId(_) => {
                write!(fmt, "{{ctor {}}}", name.display(self.0.upcast()))
            }
        }
    }
}

mod unsafe_tls {
    use super::DebugContext;
    use crate::db::HirDatabase;
    use scoped_tls::scoped_thread_local;

    scoped_thread_local!(static PROGRAM: DebugContext<'_>);

    pub(crate) fn with_current_program<R>(
        op: impl for<'a> FnOnce(Option<&'a DebugContext<'a>>) -> R,
    ) -> R {
        if PROGRAM.is_set() {
            PROGRAM.with(|prog| op(Some(prog)))
        } else {
            op(None)
        }
    }

    pub(crate) fn set_current_program<OP, R>(p: &dyn HirDatabase, op: OP) -> R
    where
        OP: FnOnce() -> R,
    {
        let ctx = DebugContext(p);
        // we're transmuting the lifetime in the DebugContext to static. This is
        // fine because we only keep the reference for the lifetime of this
        // function, *and* the only way to access the context is through
        // `with_current_program`, which hides the lifetime through the `for`
        // type.
        let static_p: &DebugContext<'static> =
            unsafe { std::mem::transmute::<&DebugContext<'_>, &DebugContext<'static>>(&ctx) };
        PROGRAM.set(static_p, op)
    }
}
