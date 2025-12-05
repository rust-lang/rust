//! Type inhabitedness logic.
use std::ops::ControlFlow::{self, Break, Continue};

use hir_def::{AdtId, EnumVariantId, ModuleId, VariantId, visibility::Visibility};
use rustc_hash::FxHashSet;
use rustc_type_ir::{
    TypeSuperVisitable, TypeVisitable, TypeVisitor,
    inherent::{AdtDef, IntoKind},
};

use crate::{
    consteval::try_const_usize,
    db::HirDatabase,
    next_solver::{
        DbInterner, EarlyBinder, GenericArgs, ParamEnv, Ty, TyKind,
        infer::{InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
};

// FIXME: Turn this into a query, it can be quite slow
/// Checks whether a type is visibly uninhabited from a particular module.
pub(crate) fn is_ty_uninhabited_from<'db>(
    infcx: &InferCtxt<'db>,
    ty: Ty<'db>,
    target_mod: ModuleId,
    env: ParamEnv<'db>,
) -> bool {
    let _p = tracing::info_span!("is_ty_uninhabited_from", ?ty).entered();
    let mut uninhabited_from = UninhabitedFrom::new(infcx, target_mod, env);
    let inhabitedness = ty.visit_with(&mut uninhabited_from);
    inhabitedness == BREAK_VISIBLY_UNINHABITED
}

// FIXME: Turn this into a query, it can be quite slow
/// Checks whether a variant is visibly uninhabited from a particular module.
pub(crate) fn is_enum_variant_uninhabited_from<'db>(
    infcx: &InferCtxt<'db>,
    variant: EnumVariantId,
    subst: GenericArgs<'db>,
    target_mod: ModuleId,
    env: ParamEnv<'db>,
) -> bool {
    let _p = tracing::info_span!("is_enum_variant_uninhabited_from").entered();

    let mut uninhabited_from = UninhabitedFrom::new(infcx, target_mod, env);
    let inhabitedness = uninhabited_from.visit_variant(variant.into(), subst);
    inhabitedness == BREAK_VISIBLY_UNINHABITED
}

struct UninhabitedFrom<'a, 'db> {
    target_mod: ModuleId,
    recursive_ty: FxHashSet<Ty<'db>>,
    // guard for preventing stack overflow in non trivial non terminating types
    max_depth: usize,
    infcx: &'a InferCtxt<'db>,
    env: ParamEnv<'db>,
}

const CONTINUE_OPAQUELY_INHABITED: ControlFlow<VisiblyUninhabited> = Continue(());
const BREAK_VISIBLY_UNINHABITED: ControlFlow<VisiblyUninhabited> = Break(VisiblyUninhabited);
#[derive(PartialEq, Eq)]
struct VisiblyUninhabited;

impl<'db> TypeVisitor<DbInterner<'db>> for UninhabitedFrom<'_, 'db> {
    type Result = ControlFlow<VisiblyUninhabited>;

    fn visit_ty(&mut self, mut ty: Ty<'db>) -> ControlFlow<VisiblyUninhabited> {
        if self.recursive_ty.contains(&ty) || self.max_depth == 0 {
            // rustc considers recursive types always inhabited. I think it is valid to consider
            // recursive types as always uninhabited, but we should do what rustc is doing.
            return CONTINUE_OPAQUELY_INHABITED;
        }
        self.recursive_ty.insert(ty);
        self.max_depth -= 1;

        if matches!(ty.kind(), TyKind::Alias(..)) {
            let mut ocx = ObligationCtxt::new(self.infcx);
            match ocx.structurally_normalize_ty(&ObligationCause::dummy(), self.env, ty) {
                Ok(it) => ty = it,
                Err(_) => return CONTINUE_OPAQUELY_INHABITED,
            }
        }

        let r = match ty.kind() {
            TyKind::Adt(adt, subst) => self.visit_adt(adt.def_id().0, subst),
            TyKind::Never => BREAK_VISIBLY_UNINHABITED,
            TyKind::Tuple(..) => ty.super_visit_with(self),
            TyKind::Array(item_ty, len) => match try_const_usize(self.infcx.interner.db, len) {
                Some(0) | None => CONTINUE_OPAQUELY_INHABITED,
                Some(1..) => item_ty.visit_with(self),
            },
            _ => CONTINUE_OPAQUELY_INHABITED,
        };
        self.recursive_ty.remove(&ty);
        self.max_depth += 1;
        r
    }
}

impl<'a, 'db> UninhabitedFrom<'a, 'db> {
    fn new(infcx: &'a InferCtxt<'db>, target_mod: ModuleId, env: ParamEnv<'db>) -> Self {
        Self { target_mod, recursive_ty: FxHashSet::default(), max_depth: 500, infcx, env }
    }

    #[inline]
    fn interner(&self) -> DbInterner<'db> {
        self.infcx.interner
    }

    #[inline]
    fn db(&self) -> &'db dyn HirDatabase {
        self.interner().db
    }

    fn visit_adt(
        &mut self,
        adt: AdtId,
        subst: GenericArgs<'db>,
    ) -> ControlFlow<VisiblyUninhabited> {
        // An ADT is uninhabited iff all its variants uninhabited.
        match adt {
            // rustc: For now, `union`s are never considered uninhabited.
            AdtId::UnionId(_) => CONTINUE_OPAQUELY_INHABITED,
            AdtId::StructId(s) => self.visit_variant(s.into(), subst),
            AdtId::EnumId(e) => {
                let enum_data = e.enum_variants(self.db());

                for &(variant, _, _) in enum_data.variants.iter() {
                    let variant_inhabitedness = self.visit_variant(variant.into(), subst);
                    match variant_inhabitedness {
                        Break(VisiblyUninhabited) => (),
                        Continue(()) => return CONTINUE_OPAQUELY_INHABITED,
                    }
                }
                BREAK_VISIBLY_UNINHABITED
            }
        }
    }

    fn visit_variant(
        &mut self,
        variant: VariantId,
        subst: GenericArgs<'db>,
    ) -> ControlFlow<VisiblyUninhabited> {
        let variant_data = variant.fields(self.db());
        let fields = variant_data.fields();
        if fields.is_empty() {
            return CONTINUE_OPAQUELY_INHABITED;
        }

        let is_enum = matches!(variant, VariantId::EnumVariantId(..));
        let field_tys = self.db().field_types(variant);
        let field_vis = if is_enum { None } else { Some(self.db().field_visibilities(variant)) };

        for (fid, _) in fields.iter() {
            self.visit_field(field_vis.as_ref().map(|it| it[fid]), &field_tys[fid], subst)?;
        }
        CONTINUE_OPAQUELY_INHABITED
    }

    fn visit_field(
        &mut self,
        vis: Option<Visibility>,
        ty: &EarlyBinder<'db, Ty<'db>>,
        subst: GenericArgs<'db>,
    ) -> ControlFlow<VisiblyUninhabited> {
        if vis.is_none_or(|it| it.is_visible_from(self.db(), self.target_mod)) {
            let ty = ty.instantiate(self.interner(), subst);
            ty.visit_with(self)
        } else {
            CONTINUE_OPAQUELY_INHABITED
        }
    }
}
