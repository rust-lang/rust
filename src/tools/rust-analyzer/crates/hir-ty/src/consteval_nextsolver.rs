//! Constant evaluation details
// FIXME(next-solver): this should get removed as things get moved to rustc_type_ir from chalk_ir
#![allow(unused)]

use base_db::Crate;
use hir_def::{
    EnumVariantId, GeneralConstId,
    expr_store::{Body, HygieneId, path::Path},
    hir::{Expr, ExprId},
    resolver::{Resolver, ValueNs},
    type_ref::LiteralConstRef,
};
use hir_expand::Lookup;
use rustc_type_ir::{
    UnevaluatedConst,
    inherent::{IntoKind, SliceLike},
};
use stdx::never;
use triomphe::Arc;

use crate::{
    ConstScalar, Interner, MemoryMap, Substitution, TraitEnvironment,
    consteval::ConstEvalError,
    db::HirDatabase,
    generics::Generics,
    infer::InferenceContext,
    next_solver::{
        Const, ConstBytes, ConstKind, DbInterner, ErrorGuaranteed, GenericArg, GenericArgs,
        ParamConst, SolverDefId, Ty, ValueConst,
        mapping::{ChalkToNextSolver, NextSolverToChalk, convert_binder_to_early_binder},
    },
};

use super::mir::{interpret_mir, lower_to_mir, pad16};

pub(crate) fn path_to_const<'a, 'g>(
    db: &'a dyn HirDatabase,
    resolver: &Resolver<'a>,
    path: &Path,
    args: impl FnOnce() -> &'g Generics,
    expected_ty: Ty<'a>,
) -> Option<Const<'a>> {
    let interner = DbInterner::new_with(db, Some(resolver.krate()), None);
    match resolver.resolve_path_in_value_ns_fully(db, path, HygieneId::ROOT) {
        Some(ValueNs::GenericParam(p)) => {
            let args = args();
            match args
                .type_or_const_param(p.into())
                .and_then(|(idx, p)| p.const_param().map(|p| (idx, p.clone())))
            {
                Some((idx, _param)) => {
                    Some(Const::new_param(interner, ParamConst { index: idx as u32, id: p }))
                }
                None => {
                    never!(
                        "Generic list doesn't contain this param: {:?}, {:?}, {:?}",
                        args,
                        path,
                        p
                    );
                    None
                }
            }
        }
        Some(ValueNs::ConstId(c)) => {
            let args = GenericArgs::new_from_iter(interner, []);
            Some(Const::new(
                interner,
                rustc_type_ir::ConstKind::Unevaluated(UnevaluatedConst::new(
                    SolverDefId::ConstId(c),
                    args,
                )),
            ))
        }
        _ => None,
    }
}

pub fn unknown_const<'db>(ty: Ty<'db>) -> Const<'db> {
    Const::new(DbInterner::conjure(), rustc_type_ir::ConstKind::Error(ErrorGuaranteed))
}

pub fn unknown_const_as_generic<'db>(ty: Ty<'db>) -> GenericArg<'db> {
    unknown_const(ty).into()
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
                ConstBytes(i.to_le_bytes()[0..size].into(), MemoryMap::default()),
            ))
        }
        LiteralConstRef::UInt(i) => {
            let size = layout.map(|it| it.size.bytes_usize()).unwrap_or(16);
            rustc_type_ir::ConstKind::Value(ValueConst::new(
                ty,
                ConstBytes(i.to_le_bytes()[0..size].into(), MemoryMap::default()),
            ))
        }
        LiteralConstRef::Bool(b) => rustc_type_ir::ConstKind::Value(ValueConst::new(
            ty,
            ConstBytes(Box::new([*b as u8]), MemoryMap::default()),
        )),
        LiteralConstRef::Char(c) => rustc_type_ir::ConstKind::Value(ValueConst::new(
            ty,
            ConstBytes((*c as u32).to_le_bytes().into(), MemoryMap::default()),
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
    let interner = DbInterner::new_with(db, None, None);
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
            let subst = unevaluated_const.args.to_chalk(interner);
            let ec = db.const_eval(c, subst, None).ok()?.to_nextsolver(interner);
            try_const_usize(db, ec)
        }
        ConstKind::Value(val) => Some(u128::from_le_bytes(pad16(&val.value.inner().0, false))),
        ConstKind::Error(_) => None,
        ConstKind::Expr(_) => None,
    }
}

pub fn try_const_isize<'db>(db: &'db dyn HirDatabase, c: &Const<'db>) -> Option<i128> {
    let interner = DbInterner::new_with(db, None, None);
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
            let subst = unevaluated_const.args.to_chalk(interner);
            let ec = db.const_eval(c, subst, None).ok()?.to_nextsolver(interner);
            try_const_isize(db, &ec)
        }
        ConstKind::Value(val) => Some(i128::from_le_bytes(pad16(&val.value.inner().0, true))),
        ConstKind::Error(_) => None,
        ConstKind::Expr(_) => None,
    }
}

pub(crate) fn const_eval_discriminant_variant(
    db: &dyn HirDatabase,
    variant_id: EnumVariantId,
) -> Result<i128, ConstEvalError> {
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
        Substitution::empty(Interner),
        db.trait_environment_for_body(def),
    )?;
    let c = interpret_mir(db, mir_body, false, None)?.0?;
    let c = c.to_nextsolver(interner);
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
pub(crate) fn eval_to_const<'db>(expr: ExprId, ctx: &mut InferenceContext<'db>) -> Const<'db> {
    let interner = DbInterner::new_with(ctx.db, None, None);
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
        return unknown_const(infer[expr].clone().to_nextsolver(interner));
    }
    if let Expr::Path(p) = &ctx.body[expr] {
        let resolver = &ctx.resolver;
        if let Some(c) = path_to_const(
            ctx.db,
            resolver,
            p,
            || ctx.generics(),
            infer[expr].to_nextsolver(interner),
        ) {
            return c;
        }
    }
    if let Ok(mir_body) = lower_to_mir(ctx.db, ctx.owner, ctx.body, &infer, expr)
        && let Ok((Ok(result), _)) = interpret_mir(ctx.db, Arc::new(mir_body), true, None)
    {
        return result.to_nextsolver(interner);
    }
    unknown_const(infer[expr].to_nextsolver(interner))
}
