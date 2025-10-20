//! Constant evaluation details

use base_db::Crate;
use chalk_ir::{BoundVar, DebruijnIndex, cast::Cast};
use hir_def::{
    expr_store::{HygieneId, path::Path},
    resolver::{Resolver, ValueNs},
    type_ref::LiteralConstRef,
};
use stdx::never;

use crate::{
    Const, ConstData, ConstScalar, ConstValue, GenericArg, Interner, MemoryMap, Substitution,
    TraitEnvironment, Ty,
    db::HirDatabase,
    generics::Generics,
    lower::ParamLoweringMode,
    next_solver::{DbInterner, mapping::ChalkToNextSolver},
    to_placeholder_idx,
};

pub(crate) fn path_to_const<'g>(
    db: &dyn HirDatabase,
    resolver: &Resolver<'_>,
    path: &Path,
    mode: ParamLoweringMode,
    args: impl FnOnce() -> &'g Generics,
    debruijn: DebruijnIndex,
    expected_ty: Ty,
) -> Option<Const> {
    match resolver.resolve_path_in_value_ns_fully(db, path, HygieneId::ROOT) {
        Some(ValueNs::GenericParam(p)) => {
            let ty = db.const_param_ty(p);
            let args = args();
            let value = match mode {
                ParamLoweringMode::Placeholder => {
                    let idx = args.type_or_const_param_idx(p.into()).unwrap();
                    ConstValue::Placeholder(to_placeholder_idx(db, p.into(), idx as u32))
                }
                ParamLoweringMode::Variable => match args.type_or_const_param_idx(p.into()) {
                    Some(it) => ConstValue::BoundVar(BoundVar::new(debruijn, it)),
                    None => {
                        never!(
                            "Generic list doesn't contain this param: {:?}, {:?}, {:?}",
                            args,
                            path,
                            p
                        );
                        return None;
                    }
                },
            };
            Some(ConstData { ty, value }.intern(Interner))
        }
        Some(ValueNs::ConstId(c)) => Some(intern_const_scalar(
            ConstScalar::UnevaluatedConst(c.into(), Substitution::empty(Interner)),
            expected_ty,
        )),
        // FIXME: With feature(adt_const_params), we also need to consider other things here, e.g. struct constructors.
        _ => None,
    }
}

pub(crate) fn unknown_const(ty: Ty) -> Const {
    ConstData {
        ty,
        value: ConstValue::Concrete(chalk_ir::ConcreteConst { interned: ConstScalar::Unknown }),
    }
    .intern(Interner)
}

pub(crate) fn unknown_const_as_generic(ty: Ty) -> GenericArg {
    unknown_const(ty).cast(Interner)
}

/// Interns a constant scalar with the given type
pub(crate) fn intern_const_scalar(value: ConstScalar, ty: Ty) -> Const {
    ConstData { ty, value: ConstValue::Concrete(chalk_ir::ConcreteConst { interned: value }) }
        .intern(Interner)
}

/// Interns a constant scalar with the given type
pub(crate) fn intern_const_ref(
    db: &dyn HirDatabase,
    value: &LiteralConstRef,
    ty: Ty,
    krate: Crate,
) -> Const {
    let interner = DbInterner::new_with(db, Some(krate), None);
    let layout = || db.layout_of_ty(ty.to_nextsolver(interner), TraitEnvironment::empty(krate));
    let bytes = match value {
        LiteralConstRef::Int(i) => {
            // FIXME: We should handle failure of layout better.
            let size = layout().map(|it| it.size.bytes_usize()).unwrap_or(16);
            ConstScalar::Bytes(i.to_le_bytes()[0..size].into(), MemoryMap::default())
        }
        LiteralConstRef::UInt(i) => {
            let size = layout().map(|it| it.size.bytes_usize()).unwrap_or(16);
            ConstScalar::Bytes(i.to_le_bytes()[0..size].into(), MemoryMap::default())
        }
        LiteralConstRef::Bool(b) => ConstScalar::Bytes(Box::new([*b as u8]), MemoryMap::default()),
        LiteralConstRef::Char(c) => {
            ConstScalar::Bytes((*c as u32).to_le_bytes().into(), MemoryMap::default())
        }
        LiteralConstRef::Unknown => ConstScalar::Unknown,
    };
    intern_const_scalar(bytes, ty)
}
