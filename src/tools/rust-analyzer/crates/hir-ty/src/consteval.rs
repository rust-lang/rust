//! Constant evaluation details

#[cfg(test)]
mod tests;

use base_db::Crate;
use hir_def::{
    ConstId, EnumVariantId, ExpressionStoreOwnerId, GenericDefId, HasModule, StaticId,
    attrs::AttrFlags,
    expr_store::{Body, ExpressionStore, HygieneId, path::Path},
    hir::{Expr, ExprId, Literal},
    resolver::{Resolver, ValueNs},
};
use hir_expand::Lookup;
use rustc_abi::Size;
use rustc_apfloat::Float;
use rustc_ast_ir::Mutability;
use rustc_type_ir::inherent::{Const as _, GenericArgs as _, IntoKind, Ty as _};
use stdx::never;

use crate::{
    ParamEnvAndCrate, Span,
    db::{AnonConstId, AnonConstLoc, GeneralConstId, HirDatabase},
    display::DisplayTarget,
    generics::Generics,
    mir::{MirEvalError, MirLowerError, pad16},
    next_solver::{
        Allocation, Const, ConstKind, Consts, DbInterner, DefaultAny, GenericArgs, ParamConst,
        ScalarInt, StoredAllocation, StoredEarlyBinder, StoredGenericArgs, Ty, TyKind,
        UnevaluatedConst, ValTreeKind, default_types,
    },
    traits::StoredParamEnvAndCrate,
};

use super::mir::interpret_mir;

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
fn intern_const_ref<'db>(
    interner: DbInterner<'db>,
    value: &Literal,
    ty: Ty<'db>,
) -> Result<Const<'db>, CreateConstError<'db>> {
    let Ok(data_layout) = interner.db.target_data_layout(interner.expect_crate()) else {
        return Ok(Const::error(interner));
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
            return Ok(Const::error(interner));
        }
        _ => {
            never!("mismatching type for literal");
            let actual = literal_ty(
                interner,
                value,
                |types| types.types.i32,
                |types| types.types.u32,
                |types| types.types.f64,
            );
            return Err(CreateConstError::TypeMismatch { actual });
        }
    };
    Ok(Const::new_valtree(interner, ty, valtree))
}

pub(crate) fn literal_ty<'db>(
    interner: DbInterner<'db>,
    value: &Literal,
    default_int: impl FnOnce(&DefaultAny<'db>) -> Ty<'db>,
    default_uint: impl FnOnce(&DefaultAny<'db>) -> Ty<'db>,
    default_float: impl FnOnce(&DefaultAny<'db>) -> Ty<'db>,
) -> Ty<'db> {
    let types = interner.default_types();
    match value {
        Literal::Bool(..) => types.types.bool,
        Literal::String(..) => types.types.static_str_ref,
        Literal::ByteString(bs) => {
            let byte_type = types.types.u8;
            let array_type = Ty::new_array(interner, byte_type, bs.len() as u128);
            Ty::new_ref(interner, types.regions.statik, array_type, Mutability::Not)
        }
        Literal::CString(..) => Ty::new_ref(
            interner,
            types.regions.statik,
            interner.lang_items().CStr.map_or(types.types.error, |strukt| {
                Ty::new_adt(interner, strukt.into(), types.empty.generic_args)
            }),
            Mutability::Not,
        ),
        Literal::Char(..) => types.types.char,
        Literal::Int(_v, ty) => match ty {
            Some(int_ty) => match int_ty {
                hir_def::builtin_type::BuiltinInt::Isize => types.types.isize,
                hir_def::builtin_type::BuiltinInt::I8 => types.types.i8,
                hir_def::builtin_type::BuiltinInt::I16 => types.types.i16,
                hir_def::builtin_type::BuiltinInt::I32 => types.types.i32,
                hir_def::builtin_type::BuiltinInt::I64 => types.types.i64,
                hir_def::builtin_type::BuiltinInt::I128 => types.types.i128,
            },
            None => default_int(types),
        },
        Literal::Uint(_v, ty) => match ty {
            Some(int_ty) => match int_ty {
                hir_def::builtin_type::BuiltinUint::Usize => types.types.usize,
                hir_def::builtin_type::BuiltinUint::U8 => types.types.u8,
                hir_def::builtin_type::BuiltinUint::U16 => types.types.u16,
                hir_def::builtin_type::BuiltinUint::U32 => types.types.u32,
                hir_def::builtin_type::BuiltinUint::U64 => types.types.u64,
                hir_def::builtin_type::BuiltinUint::U128 => types.types.u128,
            },
            None => default_uint(types),
        },
        Literal::Float(_v, ty) => match ty {
            Some(float_ty) => match float_ty {
                hir_def::builtin_type::BuiltinFloat::F16 => types.types.f16,
                hir_def::builtin_type::BuiltinFloat::F32 => types.types.f32,
                hir_def::builtin_type::BuiltinFloat::F64 => types.types.f64,
                hir_def::builtin_type::BuiltinFloat::F128 => types.types.f128,
            },
            None => default_float(types),
        },
    }
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
            GeneralConstId::AnonConstId(id) => {
                let subst = unevaluated_const.args;
                let ec = db.anon_const_eval(id, subst, None).ok()?;
                Some(allocation_as_usize(ec))
            }
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

pub fn try_const_isize<'db>(db: &'db dyn HirDatabase, c: Const<'db>) -> Option<i128> {
    match c.kind() {
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
            GeneralConstId::AnonConstId(id) => {
                let subst = unevaluated_const.args;
                let ec = db.anon_const_eval(id, subst, None).ok()?;
                Some(allocation_as_isize(ec))
            }
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

#[derive(Debug)]
pub(crate) enum CreateConstError<'db> {
    UsedForbiddenParam,
    ResolveToNonConst,
    DoesNotResolve,
    ConstHasGenerics,
    UnderscoreExpr,
    TypeMismatch {
        #[expect(unused, reason = "will need this for diagnostics")]
        actual: Ty<'db>,
    },
}

pub(crate) fn path_to_const<'a, 'db>(
    db: &'db dyn HirDatabase,
    resolver: &Resolver<'db>,
    generics: &dyn Fn() -> &'a Generics<'db>,
    forbid_params_after: Option<u32>,
    path: &Path,
) -> Result<Const<'db>, CreateConstError<'db>> {
    let interner = DbInterner::new_no_crate(db);
    let resolution = resolver
        .resolve_path_in_value_ns_fully(db, path, HygieneId::ROOT)
        .ok_or(CreateConstError::DoesNotResolve)?;
    let no_generics = |def| crate::generics::generics(db, def).has_no_params();
    let konst = match resolution {
        ValueNs::ConstId(id) if no_generics(id.into()) => GeneralConstId::ConstId(id),
        ValueNs::StaticId(id) => GeneralConstId::StaticId(id),
        ValueNs::ConstId(_) => return Err(CreateConstError::ConstHasGenerics),
        ValueNs::GenericParam(param) => {
            let index = generics().type_or_const_param_idx(param.into());
            if forbid_params_after.is_some_and(|forbid_after| index >= forbid_after) {
                return Err(CreateConstError::UsedForbiddenParam);
            }
            return Ok(Const::new_param(interner, ParamConst { id: param, index }));
        }
        // These are not valid as consts.
        // FIXME: Report an error?
        ValueNs::ImplSelf(_)
        | ValueNs::LocalBinding(_)
        | ValueNs::FunctionId(_)
        | ValueNs::StructId(_)
        | ValueNs::EnumVariantId(_) => return Err(CreateConstError::ResolveToNonConst),
    };
    let args = GenericArgs::empty(interner);
    Ok(Const::new_unevaluated(interner, UnevaluatedConst { def: konst.into(), args }))
}

pub(crate) fn create_anon_const<'a, 'db>(
    interner: DbInterner<'db>,
    owner: ExpressionStoreOwnerId,
    store: &ExpressionStore,
    expr: ExprId,
    resolver: &Resolver<'db>,
    expected_ty: Ty<'db>,
    generics: &dyn Fn() -> &'a Generics<'db>,
    create_var: Option<&mut dyn FnMut(Span) -> Const<'db>>,
    forbid_params_after: Option<u32>,
) -> Result<Const<'db>, CreateConstError<'db>> {
    match &store[expr] {
        Expr::Literal(literal) => intern_const_ref(interner, literal, expected_ty),
        Expr::Underscore => match create_var {
            Some(create_var) => Ok(create_var(expr.into())),
            None => Err(CreateConstError::UnderscoreExpr),
        },
        Expr::Path(path)
            if let konst =
                path_to_const(interner.db, resolver, generics, forbid_params_after, path)
                && !matches!(
                    konst,
                    Err(CreateConstError::DoesNotResolve | CreateConstError::ConstHasGenerics)
                ) =>
        {
            konst
        }
        _ => {
            let allow_using_generic_params = forbid_params_after.is_none();
            let konst = AnonConstId::new(
                interner.db,
                AnonConstLoc {
                    owner,
                    expr,
                    ty: StoredEarlyBinder::bind(expected_ty.store()),
                    allow_using_generic_params,
                },
            );
            let args = if allow_using_generic_params {
                GenericArgs::identity_for_item(interner, owner.generic_def(interner.db).into())
            } else {
                GenericArgs::empty(interner)
            };
            Ok(Const::new_unevaluated(
                interner,
                UnevaluatedConst { def: GeneralConstId::AnonConstId(konst).into(), args },
            ))
        }
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
        let prev_idx = loc.index(db).checked_sub(1);
        let value = match prev_idx {
            Some(prev_idx) => {
                1 + db.const_eval_discriminant(loc.parent.enum_variants(db).variants[prev_idx].0)?
            }
            _ => 0,
        };
        return Ok(value);
    }

    let repr = AttrFlags::repr(db, loc.parent.into());
    let is_signed = repr.and_then(|repr| repr.int).is_none_or(|int| int.is_signed());

    let mir_body = db.monomorphized_mir_body(
        def.into(),
        GenericArgs::empty(interner).store(),
        ParamEnvAndCrate { param_env: db.trait_environment(def.into()), krate: def.krate(db) }
            .store(),
    )?;
    let c = interpret_mir(db, mir_body, false, None)?.0?;
    let c = if is_signed { allocation_as_isize(c) } else { allocation_as_usize(c) as i128 };
    Ok(c)
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
    pub(crate) fn const_eval_query(
        db: &dyn HirDatabase,
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

pub(crate) fn anon_const_eval<'db>(
    db: &'db dyn HirDatabase,
    def: AnonConstId,
    subst: GenericArgs<'db>,
    trait_env: Option<ParamEnvAndCrate<'db>>,
) -> Result<Allocation<'db>, ConstEvalError> {
    return match anon_const_eval_query(db, def, subst.store(), trait_env.map(|env| env.store())) {
        Ok(konst) => Ok(konst.as_ref()),
        Err(err) => Err(err.clone()),
    };

    #[salsa::tracked(returns(ref), cycle_result = anon_const_eval_cycle_result)]
    pub(crate) fn anon_const_eval_query(
        db: &dyn HirDatabase,
        def: AnonConstId,
        subst: StoredGenericArgs,
        trait_env: Option<StoredParamEnvAndCrate>,
    ) -> Result<StoredAllocation, ConstEvalError> {
        let body = db.monomorphized_mir_body(
            def.into(),
            subst,
            ParamEnvAndCrate {
                param_env: db.trait_environment(def.loc(db).owner),
                krate: def.krate(db),
            }
            .store(),
        )?;
        let c = interpret_mir(db, body, false, trait_env.as_ref().map(|env| env.as_ref()))?.0?;
        Ok(c.store())
    }

    pub(crate) fn anon_const_eval_cycle_result(
        _: &dyn HirDatabase,
        _: salsa::Id,
        _: AnonConstId,
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
    pub(crate) fn const_eval_static_query(
        db: &dyn HirDatabase,
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
