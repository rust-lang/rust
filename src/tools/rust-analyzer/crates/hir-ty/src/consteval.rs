//! Constant evaluation details

use base_db::Crate;
use chalk_ir::{BoundVar, DebruijnIndex, cast::Cast};
use hir_def::{
    EnumVariantId, GeneralConstId, HasModule as _, StaticId,
    expr_store::{Body, HygieneId, path::Path},
    hir::{Expr, ExprId},
    resolver::{Resolver, ValueNs},
    type_ref::LiteralConstRef,
};
use hir_expand::Lookup;
use stdx::never;
use triomphe::Arc;

use crate::{
    Const, ConstData, ConstScalar, ConstValue, GenericArg, Interner, MemoryMap, Substitution,
    TraitEnvironment, Ty, TyBuilder, db::HirDatabase, display::DisplayTarget, generics::Generics,
    infer::InferenceContext, lower::ParamLoweringMode, to_placeholder_idx,
};

use super::mir::{MirEvalError, MirLowerError, interpret_mir, lower_to_mir, pad16};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstEvalError {
    MirLowerError(MirLowerError),
    MirEvalError(MirEvalError),
}

impl ConstEvalError {
    pub fn pretty_print(
        &self,
        f: &mut String,
        db: &dyn HirDatabase,
        span_formatter: impl Fn(span::FileId, span::TextRange) -> String,
        display_target: DisplayTarget,
    ) -> std::result::Result<(), std::fmt::Error> {
        match self {
            ConstEvalError::MirLowerError(e) => {
                e.pretty_print(f, db, span_formatter, display_target)
            }
            ConstEvalError::MirEvalError(e) => {
                e.pretty_print(f, db, span_formatter, display_target)
            }
        }
    }
}

impl From<MirLowerError> for ConstEvalError {
    fn from(value: MirLowerError) -> Self {
        match value {
            MirLowerError::ConstEvalError(_, e) => *e,
            _ => ConstEvalError::MirLowerError(value),
        }
    }
}

impl From<MirEvalError> for ConstEvalError {
    fn from(value: MirEvalError) -> Self {
        ConstEvalError::MirEvalError(value)
    }
}

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
            let value = match mode {
                ParamLoweringMode::Placeholder => {
                    ConstValue::Placeholder(to_placeholder_idx(db, p.into()))
                }
                ParamLoweringMode::Variable => {
                    let args = args();
                    match args.type_or_const_param_idx(p.into()) {
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
                    }
                }
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
    let layout = || db.layout_of_ty(ty.clone(), TraitEnvironment::empty(krate));
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
    match &c.data(Interner).value {
        chalk_ir::ConstValue::BoundVar(_) => None,
        chalk_ir::ConstValue::InferenceVar(_) => None,
        chalk_ir::ConstValue::Placeholder(_) => None,
        chalk_ir::ConstValue::Concrete(c) => match &c.interned {
            ConstScalar::Bytes(it, _) => Some(u128::from_le_bytes(pad16(it, false))),
            ConstScalar::UnevaluatedConst(c, subst) => {
                let ec = db.const_eval(*c, subst.clone(), None).ok()?;
                try_const_usize(db, &ec)
            }
            _ => None,
        },
    }
}

pub fn try_const_isize(db: &dyn HirDatabase, c: &Const) -> Option<i128> {
    match &c.data(Interner).value {
        chalk_ir::ConstValue::BoundVar(_) => None,
        chalk_ir::ConstValue::InferenceVar(_) => None,
        chalk_ir::ConstValue::Placeholder(_) => None,
        chalk_ir::ConstValue::Concrete(c) => match &c.interned {
            ConstScalar::Bytes(it, _) => Some(i128::from_le_bytes(pad16(it, true))),
            ConstScalar::UnevaluatedConst(c, subst) => {
                let ec = db.const_eval(*c, subst.clone(), None).ok()?;
                try_const_isize(db, &ec)
            }
            _ => None,
        },
    }
}

pub(crate) fn const_eval_cycle_result(
    _: &dyn HirDatabase,
    _: GeneralConstId,
    _: Substitution,
    _: Option<Arc<TraitEnvironment>>,
) -> Result<Const, ConstEvalError> {
    Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
}

pub(crate) fn const_eval_static_cycle_result(
    _: &dyn HirDatabase,
    _: StaticId,
) -> Result<Const, ConstEvalError> {
    Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
}

pub(crate) fn const_eval_discriminant_cycle_result(
    _: &dyn HirDatabase,
    _: EnumVariantId,
) -> Result<i128, ConstEvalError> {
    Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
}

pub(crate) fn const_eval_query(
    db: &dyn HirDatabase,
    def: GeneralConstId,
    subst: Substitution,
    trait_env: Option<Arc<TraitEnvironment>>,
) -> Result<Const, ConstEvalError> {
    let body = match def {
        GeneralConstId::ConstId(c) => {
            db.monomorphized_mir_body(c.into(), subst, db.trait_environment(c.into()))?
        }
        GeneralConstId::StaticId(s) => {
            let krate = s.module(db).krate();
            db.monomorphized_mir_body(s.into(), subst, TraitEnvironment::empty(krate))?
        }
    };
    let c = interpret_mir(db, body, false, trait_env)?.0?;
    Ok(c)
}

pub(crate) fn const_eval_static_query(
    db: &dyn HirDatabase,
    def: StaticId,
) -> Result<Const, ConstEvalError> {
    let body = db.monomorphized_mir_body(
        def.into(),
        Substitution::empty(Interner),
        db.trait_environment_for_body(def.into()),
    )?;
    let c = interpret_mir(db, body, false, None)?.0?;
    Ok(c)
}

pub(crate) fn const_eval_discriminant_variant(
    db: &dyn HirDatabase,
    variant_id: EnumVariantId,
) -> Result<i128, ConstEvalError> {
    let def = variant_id.into();
    let body = db.body(def);
    let loc = variant_id.lookup(db);
    if matches!(body[body.body_expr], Expr::Missing) {
        let prev_idx = loc.index.checked_sub(1);
        let value = match prev_idx {
            Some(prev_idx) => {
                1 + db.const_eval_discriminant(
                    loc.parent.enum_variants(db).variants[prev_idx as usize].0,
                )?
            }
            _ => 0,
        };
        return Ok(value);
    }

    let repr = db.enum_signature(loc.parent).repr;
    let is_signed = repr.and_then(|repr| repr.int).is_none_or(|int| int.is_signed());

    let mir_body = db.monomorphized_mir_body(
        def,
        Substitution::empty(Interner),
        db.trait_environment_for_body(def),
    )?;
    let c = interpret_mir(db, mir_body, false, None)?.0?;
    let c = if is_signed {
        try_const_isize(db, &c).unwrap()
    } else {
        try_const_usize(db, &c).unwrap() as i128
    };
    Ok(c)
}

// FIXME: Ideally constants in const eval should have separate body (issue #7434), and this function should
// get an `InferenceResult` instead of an `InferenceContext`. And we should remove `ctx.clone().resolve_all()` here
// and make this function private. See the fixme comment on `InferenceContext::resolve_all`.
pub(crate) fn eval_to_const(
    expr: ExprId,
    mode: ParamLoweringMode,
    ctx: &mut InferenceContext<'_>,
    debruijn: DebruijnIndex,
) -> Const {
    let db = ctx.db;
    let infer = ctx.clone().resolve_all();
    fn has_closure(body: &Body, expr: ExprId) -> bool {
        if matches!(body[expr], Expr::Closure { .. }) {
            return true;
        }
        let mut r = false;
        body.walk_child_exprs(expr, |idx| r |= has_closure(body, idx));
        r
    }
    if has_closure(ctx.body, expr) {
        // Type checking clousres need an isolated body (See the above FIXME). Bail out early to prevent panic.
        return unknown_const(infer[expr].clone());
    }
    if let Expr::Path(p) = &ctx.body[expr] {
        let resolver = &ctx.resolver;
        if let Some(c) =
            path_to_const(db, resolver, p, mode, || ctx.generics(), debruijn, infer[expr].clone())
        {
            return c;
        }
    }
    if let Ok(mir_body) = lower_to_mir(ctx.db, ctx.owner, ctx.body, &infer, expr)
        && let Ok((Ok(result), _)) = interpret_mir(db, Arc::new(mir_body), true, None)
    {
        return result;
    }
    unknown_const(infer[expr].clone())
}

#[cfg(test)]
mod tests;
