//! Constant evaluation details

#[cfg(test)]
mod tests;

use base_db::Crate;
use hir_def::{
    ConstId, EnumVariantId, ExpressionStoreOwnerId, GeneralConstId, GenericDefId, HasModule,
    StaticId,
    attrs::AttrFlags,
    expr_store::{Body, ExpressionStore},
    hir::{Expr, ExprId, Literal},
};
use hir_expand::Lookup;
use rustc_abi::Size;
use rustc_apfloat::Float;
use rustc_type_ir::inherent::IntoKind;
use stdx::never;
use triomphe::Arc;

use crate::{
    LifetimeElisionKind, ParamEnvAndCrate, TyLoweringContext,
    db::HirDatabase,
    display::DisplayTarget,
    infer::InferenceContext,
    mir::{MirEvalError, MirLowerError, pad16},
    next_solver::{
        Allocation, Const, ConstKind, Consts, DbInterner, ErrorGuaranteed, GenericArg, GenericArgs,
        ScalarInt, StoredAllocation, StoredGenericArgs, Ty, TyKind, ValTreeKind, default_types,
    },
    traits::StoredParamEnvAndCrate,
};

use super::mir::{interpret_mir, lower_body_to_mir};

pub fn unknown_const<'db>(_ty: Ty<'db>) -> Const<'db> {
    Const::new(DbInterner::conjure(), rustc_type_ir::ConstKind::Error(ErrorGuaranteed))
}

pub fn unknown_const_as_generic<'db>(ty: Ty<'db>) -> GenericArg<'db> {
    unknown_const(ty).into()
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

/// Interns a constant scalar with the given type
pub fn intern_const_ref<'a>(
    db: &'a dyn HirDatabase,
    value: &Literal,
    ty: Ty<'a>,
    krate: Crate,
) -> Const<'a> {
    let interner = DbInterner::new_no_crate(db);
    let Ok(data_layout) = db.target_data_layout(krate) else {
        return Const::error(interner);
    };
    let valtree = match (ty.kind(), value) {
        (TyKind::Uint(uint), Literal::Uint(value, _)) => {
            let size = uint.bit_width().map(Size::from_bits).unwrap_or(data_layout.pointer_size());
            let scalar = ScalarInt::try_from_uint(*value, size).unwrap();
            ValTreeKind::Leaf(scalar)
        }
        (TyKind::Uint(uint), Literal::Int(value, _)) => {
            // `Literal::Int` is the default, so we also need to account for the type being uint.
            let size = uint.bit_width().map(Size::from_bits).unwrap_or(data_layout.pointer_size());
            let scalar = ScalarInt::try_from_uint(*value as u128, size).unwrap();
            ValTreeKind::Leaf(scalar)
        }
        (TyKind::Int(int), Literal::Int(value, _)) => {
            let size = int.bit_width().map(Size::from_bits).unwrap_or(data_layout.pointer_size());
            let scalar = ScalarInt::try_from_int(*value, size).unwrap();
            ValTreeKind::Leaf(scalar)
        }
        (TyKind::Bool, Literal::Bool(value)) => ValTreeKind::Leaf(ScalarInt::from(*value)),
        (TyKind::Char, Literal::Char(value)) => ValTreeKind::Leaf(ScalarInt::from(*value)),
        (TyKind::Float(float), Literal::Float(value, _)) => {
            let size = Size::from_bits(float.bit_width());
            let value = match float {
                rustc_ast_ir::FloatTy::F16 => value.to_f16().to_bits(),
                rustc_ast_ir::FloatTy::F32 => value.to_f32().to_bits(),
                rustc_ast_ir::FloatTy::F64 => value.to_f64().to_bits(),
                rustc_ast_ir::FloatTy::F128 => value.to_f128().to_bits(),
            };
            let scalar = ScalarInt::try_from_uint(value, size).unwrap();
            ValTreeKind::Leaf(scalar)
        }
        (_, Literal::String(value)) => {
            let u8_values = &interner.default_types().consts.u8_values;
            ValTreeKind::Branch(Consts::new_from_iter(
                interner,
                value.as_str().as_bytes().iter().map(|&byte| u8_values[usize::from(byte)]),
            ))
        }
        (_, Literal::ByteString(value)) => {
            let u8_values = &interner.default_types().consts.u8_values;
            ValTreeKind::Branch(Consts::new_from_iter(
                interner,
                value.iter().map(|&byte| u8_values[usize::from(byte)]),
            ))
        }
        (_, Literal::CString(_)) => {
            // FIXME:
            return Const::error(interner);
        }
        _ => {
            never!("mismatching type for literal");
            return Const::error(interner);
        }
    };
    Const::new_valtree(interner, ty, valtree)
}

/// Interns a possibly-unknown target usize
pub fn usize_const<'db>(db: &'db dyn HirDatabase, value: Option<u128>, krate: Crate) -> Const<'db> {
    let interner = DbInterner::new_no_crate(db);
    let value = match value {
        Some(value) => value,
        None => {
            return Const::error(interner);
        }
    };
    let Ok(data_layout) = db.target_data_layout(krate) else {
        return Const::error(interner);
    };
    let usize_ty = interner.default_types().types.usize;
    let scalar = ScalarInt::try_from_uint(value, data_layout.pointer_size()).unwrap();
    Const::new_valtree(interner, usize_ty, ValTreeKind::Leaf(scalar))
}

pub fn allocation_as_usize(ec: Allocation<'_>) -> u128 {
    u128::from_le_bytes(pad16(&ec.memory, false))
}

pub fn try_const_usize<'db>(db: &'db dyn HirDatabase, c: Const<'db>) -> Option<u128> {
    match c.kind() {
        ConstKind::Param(_) => None,
        ConstKind::Infer(_) => None,
        ConstKind::Bound(_, _) => None,
        ConstKind::Placeholder(_) => None,
        ConstKind::Unevaluated(unevaluated_const) => match unevaluated_const.def.0 {
            GeneralConstId::ConstId(id) => {
                let subst = unevaluated_const.args;
                let ec = db.const_eval(id, subst, None).ok()?;
                Some(allocation_as_usize(ec))
            }
            GeneralConstId::StaticId(id) => {
                let ec = db.const_eval_static(id).ok()?;
                Some(allocation_as_usize(ec))
            }
            GeneralConstId::AnonConstId(_) => None,
        },
        ConstKind::Value(val) => {
            if val.ty == default_types(db).types.usize {
                Some(val.value.inner().to_leaf().to_uint_unchecked())
            } else {
                None
            }
        }
        ConstKind::Error(_) => None,
        ConstKind::Expr(_) => None,
    }
}

pub fn allocation_as_isize(ec: Allocation<'_>) -> i128 {
    i128::from_le_bytes(pad16(&ec.memory, true))
}

pub fn try_const_isize<'db>(db: &'db dyn HirDatabase, c: &Const<'db>) -> Option<i128> {
    match (*c).kind() {
        ConstKind::Param(_) => None,
        ConstKind::Infer(_) => None,
        ConstKind::Bound(_, _) => None,
        ConstKind::Placeholder(_) => None,
        ConstKind::Unevaluated(unevaluated_const) => match unevaluated_const.def.0 {
            GeneralConstId::ConstId(id) => {
                let subst = unevaluated_const.args;
                let ec = db.const_eval(id, subst, None).ok()?;
                Some(allocation_as_isize(ec))
            }
            GeneralConstId::StaticId(id) => {
                let ec = db.const_eval_static(id).ok()?;
                Some(allocation_as_isize(ec))
            }
            GeneralConstId::AnonConstId(_) => None,
        },
        ConstKind::Value(val) => {
            if val.ty == default_types(db).types.isize {
                Some(val.value.inner().to_leaf().to_int_unchecked())
            } else {
                None
            }
        }
        ConstKind::Error(_) => None,
        ConstKind::Expr(_) => None,
    }
}

pub(crate) fn const_eval_discriminant_variant(
    db: &dyn HirDatabase,
    variant_id: EnumVariantId,
) -> Result<i128, ConstEvalError> {
    let interner = DbInterner::new_no_crate(db);
    let def = variant_id.into();
    let body = Body::of(db, def);
    let loc = variant_id.lookup(db);
    if matches!(body[body.root_expr()], Expr::Missing) {
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

    let repr = AttrFlags::repr(db, loc.parent.into());
    let is_signed = repr.and_then(|repr| repr.int).is_none_or(|int| int.is_signed());

    let mir_body = db.monomorphized_mir_body(
        def,
        GenericArgs::empty(interner).store(),
        ParamEnvAndCrate { param_env: db.trait_environment(def.into()), krate: def.krate(db) }
            .store(),
    )?;
    let c = interpret_mir(db, mir_body, false, None)?.0?;
    let c = if is_signed { allocation_as_isize(c) } else { allocation_as_usize(c) as i128 };
    Ok(c)
}

// FIXME: Ideally constants in const eval should have separate body (issue #7434), and this function should
// get an `InferenceResult` instead of an `InferenceContext`. And we should remove `ctx.clone().resolve_all()` here
// and make this function private. See the fixme comment on `InferenceContext::resolve_all`.
pub(crate) fn eval_to_const<'db>(expr: ExprId, ctx: &mut InferenceContext<'_, 'db>) -> Const<'db> {
    let infer = ctx.fixme_resolve_all_clone();
    fn has_closure(store: &ExpressionStore, expr: ExprId) -> bool {
        if matches!(store[expr], Expr::Closure { .. }) {
            return true;
        }
        let mut r = false;
        store.walk_child_exprs(expr, |idx| r |= has_closure(store, idx));
        r
    }
    if has_closure(ctx.store, expr) {
        // Type checking clousres need an isolated body (See the above FIXME). Bail out early to prevent panic.
        return Const::error(ctx.interner());
    }
    if let Expr::Path(p) = &ctx.store[expr] {
        let mut ctx = TyLoweringContext::new(
            ctx.db,
            &ctx.resolver,
            ctx.store,
            ctx.generic_def,
            LifetimeElisionKind::Infer,
        );
        if let Some(c) = ctx.path_to_const(p) {
            return c;
        }
    }
    if let Some(body_owner) = ctx.owner.as_def_with_body()
        && let Ok(mir_body) =
            lower_body_to_mir(ctx.db, body_owner, Body::of(ctx.db, body_owner), &infer, expr)
        && let Ok((Ok(result), _)) = interpret_mir(ctx.db, Arc::new(mir_body), true, None)
    {
        return Const::new_from_allocation(
            ctx.interner(),
            &result,
            ParamEnvAndCrate { param_env: ctx.table.param_env, krate: ctx.resolver.krate() },
        );
    }
    Const::error(ctx.interner())
}

pub(crate) fn const_eval_discriminant_cycle_result(
    _: &dyn HirDatabase,
    _: salsa::Id,
    _: EnumVariantId,
) -> Result<i128, ConstEvalError> {
    Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
}

pub(crate) fn const_eval<'db>(
    db: &'db dyn HirDatabase,
    def: ConstId,
    subst: GenericArgs<'db>,
    trait_env: Option<ParamEnvAndCrate<'db>>,
) -> Result<Allocation<'db>, ConstEvalError> {
    return match const_eval_query(db, def, subst.store(), trait_env.map(|env| env.store())) {
        Ok(konst) => Ok(konst.as_ref()),
        Err(err) => Err(err.clone()),
    };

    #[salsa::tracked(returns(ref), cycle_result = const_eval_cycle_result)]
    pub(crate) fn const_eval_query<'db>(
        db: &'db dyn HirDatabase,
        def: ConstId,
        subst: StoredGenericArgs,
        trait_env: Option<StoredParamEnvAndCrate>,
    ) -> Result<StoredAllocation, ConstEvalError> {
        let body = db.monomorphized_mir_body(
            def.into(),
            subst,
            ParamEnvAndCrate {
                param_env: db
                    .trait_environment(ExpressionStoreOwnerId::from(GenericDefId::from(def))),
                krate: def.krate(db),
            }
            .store(),
        )?;
        let c = interpret_mir(db, body, false, trait_env.as_ref().map(|env| env.as_ref()))?.0?;
        Ok(c.store())
    }

    pub(crate) fn const_eval_cycle_result(
        _: &dyn HirDatabase,
        _: salsa::Id,
        _: ConstId,
        _: StoredGenericArgs,
        _: Option<StoredParamEnvAndCrate>,
    ) -> Result<StoredAllocation, ConstEvalError> {
        Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
    }
}

pub(crate) fn const_eval_static<'db>(
    db: &'db dyn HirDatabase,
    def: StaticId,
) -> Result<Allocation<'db>, ConstEvalError> {
    return match const_eval_static_query(db, def) {
        Ok(konst) => Ok(konst.as_ref()),
        Err(err) => Err(err.clone()),
    };

    #[salsa::tracked(returns(ref), cycle_result = const_eval_static_cycle_result)]
    pub(crate) fn const_eval_static_query<'db>(
        db: &'db dyn HirDatabase,
        def: StaticId,
    ) -> Result<StoredAllocation, ConstEvalError> {
        let interner = DbInterner::new_no_crate(db);
        let body = db.monomorphized_mir_body(
            def.into(),
            GenericArgs::empty(interner).store(),
            ParamEnvAndCrate {
                param_env: db
                    .trait_environment(ExpressionStoreOwnerId::from(GenericDefId::from(def))),
                krate: def.krate(db),
            }
            .store(),
        )?;
        let c = interpret_mir(db, body, false, None)?.0?;
        Ok(c.store())
    }

    pub(crate) fn const_eval_static_cycle_result(
        _: &dyn HirDatabase,
        _: salsa::Id,
        _: StaticId,
    ) -> Result<StoredAllocation, ConstEvalError> {
        Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
    }
}
