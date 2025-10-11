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
    TraitEnvironment, Ty, TyBuilder,
    db::HirDatabase,
    generics::Generics,
    lower::ParamLoweringMode,
    next_solver::{
        DbInterner,
        mapping::{ChalkToNextSolver, NextSolverToChalk},
    },
    to_placeholder_idx,
};

use super::mir::pad16;

/// Extension trait for [`Const`]
pub trait ConstExt {
    /// Is a [`Const`] unknown?
    fn is_unknown(&self) -> bool;
}

impl ConstExt for Const {
    fn is_unknown(&self) -> bool {
        match self.data(Interner).value {
            // interned Unknown
            chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst {
                interned: ConstScalar::Unknown,
            }) => true,

            // interned concrete anything else
            chalk_ir::ConstValue::Concrete(..) => false,

            _ => {
                tracing::error!(
                    "is_unknown was called on a non-concrete constant value! {:?}",
                    self
                );
                true
            }
        }
    }
}

pub fn path_to_const<'g>(
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

pub fn unknown_const(ty: Ty) -> Const {
    ConstData {
        ty,
        value: ConstValue::Concrete(chalk_ir::ConcreteConst { interned: ConstScalar::Unknown }),
    }
    .intern(Interner)
}

pub fn unknown_const_as_generic(ty: Ty) -> GenericArg {
    unknown_const(ty).cast(Interner)
}

/// Interns a constant scalar with the given type
pub fn intern_const_scalar(value: ConstScalar, ty: Ty) -> Const {
    ConstData { ty, value: ConstValue::Concrete(chalk_ir::ConcreteConst { interned: value }) }
        .intern(Interner)
}

/// Interns a constant scalar with the given type
pub fn intern_const_ref(
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

/// Interns a possibly-unknown target usize
pub fn usize_const(db: &dyn HirDatabase, value: Option<u128>, krate: Crate) -> Const {
    intern_const_ref(
        db,
        &value.map_or(LiteralConstRef::Unknown, LiteralConstRef::UInt),
        TyBuilder::usize(),
        krate,
    )
}

pub fn try_const_usize(db: &dyn HirDatabase, c: &Const) -> Option<u128> {
    let interner = DbInterner::new_with(db, None, None);
    match &c.data(Interner).value {
        chalk_ir::ConstValue::BoundVar(_) => None,
        chalk_ir::ConstValue::InferenceVar(_) => None,
        chalk_ir::ConstValue::Placeholder(_) => None,
        chalk_ir::ConstValue::Concrete(c) => match &c.interned {
            ConstScalar::Bytes(it, _) => Some(u128::from_le_bytes(pad16(it, false))),
            ConstScalar::UnevaluatedConst(c, subst) => {
                let ec = db.const_eval(*c, subst.to_nextsolver(interner), None).ok()?;
                try_const_usize(db, &ec.to_chalk(interner))
            }
            _ => None,
        },
    }
}

pub fn try_const_isize(db: &dyn HirDatabase, c: &Const) -> Option<i128> {
    let interner = DbInterner::new_with(db, None, None);
    match &c.data(Interner).value {
        chalk_ir::ConstValue::BoundVar(_) => None,
        chalk_ir::ConstValue::InferenceVar(_) => None,
        chalk_ir::ConstValue::Placeholder(_) => None,
        chalk_ir::ConstValue::Concrete(c) => match &c.interned {
            ConstScalar::Bytes(it, _) => Some(i128::from_le_bytes(pad16(it, true))),
            ConstScalar::UnevaluatedConst(c, subst) => {
                let ec = db.const_eval(*c, subst.to_nextsolver(interner), None).ok()?;
                try_const_isize(db, &ec.to_chalk(interner))
            }
            _ => None,
        },
    }
}
