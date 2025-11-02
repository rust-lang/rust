//! Constant evaluation details

#[cfg(test)]
mod tests;

use base_db::Crate;
use hir_def::{
    EnumVariantId, GeneralConstId, HasModule, StaticId,
    expr_store::Body,
    hir::{Expr, ExprId},
    type_ref::LiteralConstRef,
};
use hir_expand::Lookup;
use rustc_type_ir::inherent::IntoKind;
use triomphe::Arc;

use crate::{
    LifetimeElisionKind, MemoryMap, TraitEnvironment, TyLoweringContext,
    db::HirDatabase,
    display::DisplayTarget,
    infer::InferenceContext,
    mir::{MirEvalError, MirLowerError},
    next_solver::{
        Const, ConstBytes, ConstKind, DbInterner, ErrorGuaranteed, GenericArg, GenericArgs,
        SolverDefId, Ty, ValueConst,
    },
};

use super::mir::{interpret_mir, lower_to_mir, pad16};

pub fn unknown_const<'db>(_ty: Ty<'db>) -> Const<'db> {
    Const::new(DbInterner::conjure(), rustc_type_ir::ConstKind::Error(ErrorGuaranteed))
}

pub fn unknown_const_as_generic<'db>(ty: Ty<'db>) -> GenericArg<'db> {
    unknown_const(ty).into()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstEvalError<'db> {
    MirLowerError(MirLowerError<'db>),
    MirEvalError(MirEvalError<'db>),
}

impl ConstEvalError<'_> {
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

impl<'db> From<MirLowerError<'db>> for ConstEvalError<'db> {
    fn from(value: MirLowerError<'db>) -> Self {
        match value {
            MirLowerError::ConstEvalError(_, e) => *e,
            _ => ConstEvalError::MirLowerError(value),
        }
    }
}

impl<'db> From<MirEvalError<'db>> for ConstEvalError<'db> {
    fn from(value: MirEvalError<'db>) -> Self {
        ConstEvalError::MirEvalError(value)
    }
}

/// Interns a constant scalar with the given type
pub fn intern_const_ref<'a>(
    db: &'a dyn HirDatabase,
    value: &LiteralConstRef,
    ty: Ty<'a>,
    krate: Crate,
) -> Const<'a> {
    let interner = DbInterner::new_with(db, Some(krate), None);
    let layout = db.layout_of_ty(ty, TraitEnvironment::empty(krate));
    let kind = match value {
        LiteralConstRef::Int(i) => {
            // FIXME: We should handle failure of layout better.
            let size = layout.map(|it| it.size.bytes_usize()).unwrap_or(16);
            rustc_type_ir::ConstKind::Value(ValueConst::new(
                ty,
                ConstBytes {
                    memory: i.to_le_bytes()[0..size].into(),
                    memory_map: MemoryMap::default(),
                },
            ))
        }
        LiteralConstRef::UInt(i) => {
            let size = layout.map(|it| it.size.bytes_usize()).unwrap_or(16);
            rustc_type_ir::ConstKind::Value(ValueConst::new(
                ty,
                ConstBytes {
                    memory: i.to_le_bytes()[0..size].into(),
                    memory_map: MemoryMap::default(),
                },
            ))
        }
        LiteralConstRef::Bool(b) => rustc_type_ir::ConstKind::Value(ValueConst::new(
            ty,
            ConstBytes { memory: Box::new([*b as u8]), memory_map: MemoryMap::default() },
        )),
        LiteralConstRef::Char(c) => rustc_type_ir::ConstKind::Value(ValueConst::new(
            ty,
            ConstBytes {
                memory: (*c as u32).to_le_bytes().into(),
                memory_map: MemoryMap::default(),
            },
        )),
        LiteralConstRef::Unknown => rustc_type_ir::ConstKind::Error(ErrorGuaranteed),
    };
    Const::new(interner, kind)
}

/// Interns a possibly-unknown target usize
pub fn usize_const<'db>(db: &'db dyn HirDatabase, value: Option<u128>, krate: Crate) -> Const<'db> {
    intern_const_ref(
        db,
        &value.map_or(LiteralConstRef::Unknown, LiteralConstRef::UInt),
        Ty::new_uint(DbInterner::new_with(db, Some(krate), None), rustc_type_ir::UintTy::Usize),
        krate,
    )
}

pub fn try_const_usize<'db>(db: &'db dyn HirDatabase, c: Const<'db>) -> Option<u128> {
    match c.kind() {
        ConstKind::Param(_) => None,
        ConstKind::Infer(_) => None,
        ConstKind::Bound(_, _) => None,
        ConstKind::Placeholder(_) => None,
        ConstKind::Unevaluated(unevaluated_const) => {
            let c = match unevaluated_const.def {
                SolverDefId::ConstId(id) => GeneralConstId::ConstId(id),
                SolverDefId::StaticId(id) => GeneralConstId::StaticId(id),
                _ => unreachable!(),
            };
            let subst = unevaluated_const.args;
            let ec = db.const_eval(c, subst, None).ok()?;
            try_const_usize(db, ec)
        }
        ConstKind::Value(val) => Some(u128::from_le_bytes(pad16(&val.value.inner().memory, false))),
        ConstKind::Error(_) => None,
        ConstKind::Expr(_) => None,
    }
}

pub fn try_const_isize<'db>(db: &'db dyn HirDatabase, c: &Const<'db>) -> Option<i128> {
    match (*c).kind() {
        ConstKind::Param(_) => None,
        ConstKind::Infer(_) => None,
        ConstKind::Bound(_, _) => None,
        ConstKind::Placeholder(_) => None,
        ConstKind::Unevaluated(unevaluated_const) => {
            let c = match unevaluated_const.def {
                SolverDefId::ConstId(id) => GeneralConstId::ConstId(id),
                SolverDefId::StaticId(id) => GeneralConstId::StaticId(id),
                _ => unreachable!(),
            };
            let subst = unevaluated_const.args;
            let ec = db.const_eval(c, subst, None).ok()?;
            try_const_isize(db, &ec)
        }
        ConstKind::Value(val) => Some(i128::from_le_bytes(pad16(&val.value.inner().memory, true))),
        ConstKind::Error(_) => None,
        ConstKind::Expr(_) => None,
    }
}

pub(crate) fn const_eval_discriminant_variant<'db>(
    db: &'db dyn HirDatabase,
    variant_id: EnumVariantId,
) -> Result<i128, ConstEvalError<'db>> {
    let interner = DbInterner::new_with(db, None, None);
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
        GenericArgs::new_from_iter(interner, []),
        db.trait_environment_for_body(def),
    )?;
    let c = interpret_mir(db, mir_body, false, None)?.0?;
    let c = if is_signed {
        try_const_isize(db, &c).unwrap()
    } else {
        try_const_usize(db, c).unwrap() as i128
    };
    Ok(c)
}

// FIXME: Ideally constants in const eval should have separate body (issue #7434), and this function should
// get an `InferenceResult` instead of an `InferenceContext`. And we should remove `ctx.clone().resolve_all()` here
// and make this function private. See the fixme comment on `InferenceContext::resolve_all`.
pub(crate) fn eval_to_const<'db>(expr: ExprId, ctx: &mut InferenceContext<'_, 'db>) -> Const<'db> {
    let infer = ctx.fixme_resolve_all_clone();
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
        return unknown_const(infer[expr]);
    }
    if let Expr::Path(p) = &ctx.body[expr] {
        let mut ctx = TyLoweringContext::new(
            ctx.db,
            &ctx.resolver,
            ctx.body,
            ctx.generic_def,
            LifetimeElisionKind::Infer,
        );
        if let Some(c) = ctx.path_to_const(p) {
            return c;
        }
    }
    if let Ok(mir_body) = lower_to_mir(ctx.db, ctx.owner, ctx.body, &infer, expr)
        && let Ok((Ok(result), _)) = interpret_mir(ctx.db, Arc::new(mir_body), true, None)
    {
        return result;
    }
    unknown_const(infer[expr])
}

pub(crate) fn const_eval_cycle_result<'db>(
    _: &'db dyn HirDatabase,
    _: GeneralConstId,
    _: GenericArgs<'db>,
    _: Option<Arc<TraitEnvironment<'db>>>,
) -> Result<Const<'db>, ConstEvalError<'db>> {
    Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
}

pub(crate) fn const_eval_static_cycle_result<'db>(
    _: &'db dyn HirDatabase,
    _: StaticId,
) -> Result<Const<'db>, ConstEvalError<'db>> {
    Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
}

pub(crate) fn const_eval_discriminant_cycle_result<'db>(
    _: &'db dyn HirDatabase,
    _: EnumVariantId,
) -> Result<i128, ConstEvalError<'db>> {
    Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
}

pub(crate) fn const_eval_query<'db>(
    db: &'db dyn HirDatabase,
    def: GeneralConstId,
    subst: GenericArgs<'db>,
    trait_env: Option<Arc<TraitEnvironment<'db>>>,
) -> Result<Const<'db>, ConstEvalError<'db>> {
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

pub(crate) fn const_eval_static_query<'db>(
    db: &'db dyn HirDatabase,
    def: StaticId,
) -> Result<Const<'db>, ConstEvalError<'db>> {
    let interner = DbInterner::new_with(db, None, None);
    let body = db.monomorphized_mir_body(
        def.into(),
        GenericArgs::new_from_iter(interner, []),
        db.trait_environment_for_body(def.into()),
    )?;
    let c = interpret_mir(db, body, false, None)?.0?;
    Ok(c)
}
