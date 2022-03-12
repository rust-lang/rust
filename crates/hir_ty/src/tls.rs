//! Implementation of Chalk debug helper functions using TLS.
use std::fmt;

use itertools::Itertools;

use crate::{
    chalk_db, db::HirDatabase, from_assoc_type_id, from_chalk_trait_id, mapping::from_chalk,
    CallableDefId, Interner,
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
        write!(f, "{}", name)
    }

    pub(crate) fn debug_trait_id(
        &self,
        id: chalk_db::TraitId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Result<(), fmt::Error> {
        let trait_: hir_def::TraitId = from_chalk_trait_id(id);
        let trait_data = self.0.trait_data(trait_);
        write!(fmt, "{}", trait_data.name)
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
        write!(fmt, "{}::{}", trait_data.name, type_alias_data.name)
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
        let trait_data = self.0.trait_data(trait_);
        let params = projection_ty.substitution.as_slice(Interner);
        write!(fmt, "<{:?} as {}", &params[0], trait_data.name,)?;
        if params.len() > 1 {
            write!(
                fmt,
                "<{}>",
                &params[1..].iter().format_with(", ", |x, f| f(&format_args!("{:?}", x))),
            )?;
        }
        write!(fmt, ">::{}", type_alias_data.name)
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
            CallableDefId::FunctionId(_) => write!(fmt, "{{fn {}}}", name),
            CallableDefId::StructId(_) | CallableDefId::EnumVariantId(_) => {
                write!(fmt, "{{ctor {}}}", name)
            }
        }
    }
}

mod unsafe_tls {
    use super::DebugContext;
    use crate::db::HirDatabase;
    use scoped_tls::scoped_thread_local;

    scoped_thread_local!(static PROGRAM: DebugContext);

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
            unsafe { std::mem::transmute::<&DebugContext, &DebugContext<'static>>(&ctx) };
        PROGRAM.set(static_p, op)
    }
}
