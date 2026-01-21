//! Constant evaluation details

#[cfg(test)]
mod tests;

use base_db::Crate;
use hir_def::{
    ConstId, EnumVariantId, GeneralConstId, HasModule, StaticId,
    attrs::AttrFlags,
    builtin_type::{BuiltinInt, BuiltinType, BuiltinUint},
    expr_store::Body,
    hir::{Expr, ExprId, Literal},
};
use hir_expand::Lookup;
use rustc_type_ir::inherent::IntoKind;
use triomphe::Arc;

use crate::{
    LifetimeElisionKind, MemoryMap, ParamEnvAndCrate, TyLoweringContext,
    db::HirDatabase,
    display::DisplayTarget,
    infer::InferenceContext,
    mir::{MirEvalError, MirLowerError},
    next_solver::{
        Const, ConstBytes, ConstKind, DbInterner, ErrorGuaranteed, GenericArg, GenericArgs,
        StoredConst, StoredGenericArgs, Ty, ValueConst,
    },
    traits::StoredParamEnvAndCrate,
};

use super::mir::{interpret_mir, lower_to_mir, pad16};

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
    _krate: Crate,
) -> Const<'a> {
    let interner = DbInterner::new_no_crate(db);
    let kind = match value {
        &Literal::Uint(i, builtin_ty)
            if builtin_ty.is_none() || ty.as_builtin() == builtin_ty.map(BuiltinType::Uint) =>
        {
            let memory = match ty.as_builtin() {
                Some(BuiltinType::Uint(builtin_uint)) => match builtin_uint {
                    BuiltinUint::U8 => Box::new([i as u8]) as Box<[u8]>,
                    BuiltinUint::U16 => Box::new((i as u16).to_le_bytes()),
                    BuiltinUint::U32 => Box::new((i as u32).to_le_bytes()),
                    BuiltinUint::U64 => Box::new((i as u64).to_le_bytes()),
                    BuiltinUint::U128 => Box::new((i).to_le_bytes()),
                    BuiltinUint::Usize => Box::new((i as usize).to_le_bytes()),
                },
                _ => return Const::new(interner, rustc_type_ir::ConstKind::Error(ErrorGuaranteed)),
            };
            rustc_type_ir::ConstKind::Value(ValueConst::new(
                ty,
                ConstBytes { memory, memory_map: MemoryMap::default() },
            ))
        }
        &Literal::Int(i, None)
            if ty
                .as_builtin()
                .is_some_and(|builtin_ty| matches!(builtin_ty, BuiltinType::Uint(_))) =>
        {
            let memory = match ty.as_builtin() {
                Some(BuiltinType::Uint(builtin_uint)) => match builtin_uint {
                    BuiltinUint::U8 => Box::new([i as u8]) as Box<[u8]>,
                    BuiltinUint::U16 => Box::new((i as u16).to_le_bytes()),
                    BuiltinUint::U32 => Box::new((i as u32).to_le_bytes()),
                    BuiltinUint::U64 => Box::new((i as u64).to_le_bytes()),
                    BuiltinUint::U128 => Box::new((i as u128).to_le_bytes()),
                    BuiltinUint::Usize => Box::new((i as usize).to_le_bytes()),
                },
                _ => return Const::new(interner, rustc_type_ir::ConstKind::Error(ErrorGuaranteed)),
            };
            rustc_type_ir::ConstKind::Value(ValueConst::new(
                ty,
                ConstBytes { memory, memory_map: MemoryMap::default() },
            ))
        }
        &Literal::Int(i, builtin_ty)
            if builtin_ty.is_none() || ty.as_builtin() == builtin_ty.map(BuiltinType::Int) =>
        {
            let memory = match ty.as_builtin() {
                Some(BuiltinType::Int(builtin_int)) => match builtin_int {
                    BuiltinInt::I8 => Box::new([i as u8]) as Box<[u8]>,
                    BuiltinInt::I16 => Box::new((i as i16).to_le_bytes()),
                    BuiltinInt::I32 => Box::new((i as i32).to_le_bytes()),
                    BuiltinInt::I64 => Box::new((i as i64).to_le_bytes()),
                    BuiltinInt::I128 => Box::new((i).to_le_bytes()),
                    BuiltinInt::Isize => Box::new((i as isize).to_le_bytes()),
                },
                _ => return Const::new(interner, rustc_type_ir::ConstKind::Error(ErrorGuaranteed)),
            };
            rustc_type_ir::ConstKind::Value(ValueConst::new(
                ty,
                ConstBytes { memory, memory_map: MemoryMap::default() },
            ))
        }
        Literal::Float(float_type_wrapper, builtin_float)
            if builtin_float.is_none()
                || ty.as_builtin() == builtin_float.map(BuiltinType::Float) =>
        {
            let memory = match ty.as_builtin().unwrap() {
                BuiltinType::Float(builtin_float) => match builtin_float {
                    // FIXME:
                    hir_def::builtin_type::BuiltinFloat::F16 => Box::new([0u8; 2]) as Box<[u8]>,
                    hir_def::builtin_type::BuiltinFloat::F32 => {
                        Box::new(float_type_wrapper.to_f32().to_le_bytes())
                    }
                    hir_def::builtin_type::BuiltinFloat::F64 => {
                        Box::new(float_type_wrapper.to_f64().to_le_bytes())
                    }
                    // FIXME:
                    hir_def::builtin_type::BuiltinFloat::F128 => Box::new([0; 16]),
                },
                _ => unreachable!(),
            };
            rustc_type_ir::ConstKind::Value(ValueConst::new(
                ty,
                ConstBytes { memory, memory_map: MemoryMap::default() },
            ))
        }
        Literal::Bool(b) if ty.is_bool() => rustc_type_ir::ConstKind::Value(ValueConst::new(
            ty,
            ConstBytes { memory: Box::new([*b as u8]), memory_map: MemoryMap::default() },
        )),
        Literal::Char(c) if ty.is_char() => rustc_type_ir::ConstKind::Value(ValueConst::new(
            ty,
            ConstBytes {
                memory: (*c as u32).to_le_bytes().into(),
                memory_map: MemoryMap::default(),
            },
        )),
        Literal::String(symbol) if ty.is_str() => rustc_type_ir::ConstKind::Value(ValueConst::new(
            ty,
            ConstBytes {
                memory: symbol.as_str().as_bytes().into(),
                memory_map: MemoryMap::default(),
            },
        )),
        Literal::ByteString(items) if ty.as_slice().is_some_and(|ty| ty.is_u8()) => {
            rustc_type_ir::ConstKind::Value(ValueConst::new(
                ty,
                ConstBytes { memory: items.clone(), memory_map: MemoryMap::default() },
            ))
        }
        // FIXME
        Literal::CString(_items) => rustc_type_ir::ConstKind::Error(ErrorGuaranteed),
        _ => rustc_type_ir::ConstKind::Error(ErrorGuaranteed),
    };
    Const::new(interner, kind)
}

/// Interns a possibly-unknown target usize
pub fn usize_const<'db>(db: &'db dyn HirDatabase, value: Option<u128>, krate: Crate) -> Const<'db> {
    intern_const_ref(
        db,
        &match value {
            Some(value) => Literal::Uint(value, Some(BuiltinUint::Usize)),
            None => {
                return Const::new(
                    DbInterner::new_no_crate(db),
                    rustc_type_ir::ConstKind::Error(ErrorGuaranteed),
                );
            }
        },
        Ty::new_uint(DbInterner::new_no_crate(db), rustc_type_ir::UintTy::Usize),
        krate,
    )
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
                try_const_usize(db, ec)
            }
            GeneralConstId::StaticId(id) => {
                let ec = db.const_eval_static(id).ok()?;
                try_const_usize(db, ec)
            }
        },
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
        ConstKind::Unevaluated(unevaluated_const) => match unevaluated_const.def.0 {
            GeneralConstId::ConstId(id) => {
                let subst = unevaluated_const.args;
                let ec = db.const_eval(id, subst, None).ok()?;
                try_const_isize(db, &ec)
            }
            GeneralConstId::StaticId(id) => {
                let ec = db.const_eval_static(id).ok()?;
                try_const_isize(db, &ec)
            }
        },
        ConstKind::Value(val) => Some(i128::from_le_bytes(pad16(&val.value.inner().memory, true))),
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

    let repr = AttrFlags::repr(db, loc.parent.into());
    let is_signed = repr.and_then(|repr| repr.int).is_none_or(|int| int.is_signed());

    let mir_body = db.monomorphized_mir_body(
        def,
        GenericArgs::empty(interner).store(),
        ParamEnvAndCrate { param_env: db.trait_environment_for_body(def), krate: def.krate(db) }
            .store(),
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
        return Const::error(ctx.interner());
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
) -> Result<Const<'db>, ConstEvalError> {
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
    ) -> Result<StoredConst, ConstEvalError> {
        let body = db.monomorphized_mir_body(
            def.into(),
            subst,
            ParamEnvAndCrate { param_env: db.trait_environment(def.into()), krate: def.krate(db) }
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
    ) -> Result<StoredConst, ConstEvalError> {
        Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
    }
}

pub(crate) fn const_eval_static<'db>(
    db: &'db dyn HirDatabase,
    def: StaticId,
) -> Result<Const<'db>, ConstEvalError> {
    return match const_eval_static_query(db, def) {
        Ok(konst) => Ok(konst.as_ref()),
        Err(err) => Err(err.clone()),
    };

    #[salsa::tracked(returns(ref), cycle_result = const_eval_static_cycle_result)]
    pub(crate) fn const_eval_static_query<'db>(
        db: &'db dyn HirDatabase,
        def: StaticId,
    ) -> Result<StoredConst, ConstEvalError> {
        let interner = DbInterner::new_no_crate(db);
        let body = db.monomorphized_mir_body(
            def.into(),
            GenericArgs::empty(interner).store(),
            ParamEnvAndCrate {
                param_env: db.trait_environment_for_body(def.into()),
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
    ) -> Result<StoredConst, ConstEvalError> {
        Err(ConstEvalError::MirLowerError(MirLowerError::Loop))
    }
}
