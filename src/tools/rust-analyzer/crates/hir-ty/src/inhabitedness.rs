//! Type inhabitedness logic.
use std::ops::ControlFlow::{self, Break, Continue};

use chalk_ir::{
    DebruijnIndex,
    visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
};
use hir_def::{AdtId, EnumVariantId, ModuleId, VariantId, visibility::Visibility};
use rustc_hash::FxHashSet;
use triomphe::Arc;

use crate::{
    AliasTy, Binders, Interner, Substitution, TraitEnvironment, Ty, TyKind,
    consteval::try_const_usize, db::HirDatabase,
};

// FIXME: Turn this into a query, it can be quite slow
/// Checks whether a type is visibly uninhabited from a particular module.
pub(crate) fn is_ty_uninhabited_from(
    db: &dyn HirDatabase,
    ty: &Ty,
    target_mod: ModuleId,
    env: Arc<TraitEnvironment>,
) -> bool {
    let _p = tracing::info_span!("is_ty_uninhabited_from", ?ty).entered();
    let mut uninhabited_from =
        UninhabitedFrom { target_mod, db, max_depth: 500, recursive_ty: FxHashSet::default(), env };
    let inhabitedness = ty.visit_with(&mut uninhabited_from, DebruijnIndex::INNERMOST);
    inhabitedness == BREAK_VISIBLY_UNINHABITED
}

// FIXME: Turn this into a query, it can be quite slow
/// Checks whether a variant is visibly uninhabited from a particular module.
pub(crate) fn is_enum_variant_uninhabited_from(
    db: &dyn HirDatabase,
    variant: EnumVariantId,
    subst: &Substitution,
    target_mod: ModuleId,
    env: Arc<TraitEnvironment>,
) -> bool {
    let _p = tracing::info_span!("is_enum_variant_uninhabited_from").entered();

    let mut uninhabited_from =
        UninhabitedFrom { target_mod, db, max_depth: 500, recursive_ty: FxHashSet::default(), env };
    let inhabitedness = uninhabited_from.visit_variant(variant.into(), subst);
    inhabitedness == BREAK_VISIBLY_UNINHABITED
}

struct UninhabitedFrom<'a> {
    target_mod: ModuleId,
    recursive_ty: FxHashSet<Ty>,
    // guard for preventing stack overflow in non trivial non terminating types
    max_depth: usize,
    db: &'a dyn HirDatabase,
    env: Arc<TraitEnvironment>,
}

const CONTINUE_OPAQUELY_INHABITED: ControlFlow<VisiblyUninhabited> = Continue(());
const BREAK_VISIBLY_UNINHABITED: ControlFlow<VisiblyUninhabited> = Break(VisiblyUninhabited);
#[derive(PartialEq, Eq)]
struct VisiblyUninhabited;

impl TypeVisitor<Interner> for UninhabitedFrom<'_> {
    type BreakTy = VisiblyUninhabited;

    fn as_dyn(&mut self) -> &mut dyn TypeVisitor<Interner, BreakTy = VisiblyUninhabited> {
        self
    }

    fn visit_ty(
        &mut self,
        ty: &Ty,
        outer_binder: DebruijnIndex,
    ) -> ControlFlow<VisiblyUninhabited> {
        if self.recursive_ty.contains(ty) || self.max_depth == 0 {
            // rustc considers recursive types always inhabited. I think it is valid to consider
            // recursive types as always uninhabited, but we should do what rustc is doing.
            return CONTINUE_OPAQUELY_INHABITED;
        }
        self.recursive_ty.insert(ty.clone());
        self.max_depth -= 1;
        let r = match ty.kind(Interner) {
            TyKind::Adt(adt, subst) => self.visit_adt(adt.0, subst),
            TyKind::Never => BREAK_VISIBLY_UNINHABITED,
            TyKind::Tuple(..) => ty.super_visit_with(self, outer_binder),
            TyKind::Array(item_ty, len) => match try_const_usize(self.db, len) {
                Some(0) | None => CONTINUE_OPAQUELY_INHABITED,
                Some(1..) => item_ty.super_visit_with(self, outer_binder),
            },
            TyKind::Alias(AliasTy::Projection(projection)) => {
                // FIXME: I think this currently isn't used for monomorphized bodies, so there is no need to handle
                // `TyKind::AssociatedType`, but perhaps in the future it will.
                let normalized = self.db.normalize_projection(projection.clone(), self.env.clone());
                self.visit_ty(&normalized, outer_binder)
            }
            _ => CONTINUE_OPAQUELY_INHABITED,
        };
        self.recursive_ty.remove(ty);
        self.max_depth += 1;
        r
    }

    fn interner(&self) -> Interner {
        Interner
    }
}

impl UninhabitedFrom<'_> {
    fn visit_adt(&mut self, adt: AdtId, subst: &Substitution) -> ControlFlow<VisiblyUninhabited> {
        // An ADT is uninhabited iff all its variants uninhabited.
        match adt {
            // rustc: For now, `union`s are never considered uninhabited.
            AdtId::UnionId(_) => CONTINUE_OPAQUELY_INHABITED,
            AdtId::StructId(s) => self.visit_variant(s.into(), subst),
            AdtId::EnumId(e) => {
                let enum_data = self.db.enum_variants(e);

                for &(variant, _) in enum_data.variants.iter() {
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
        subst: &Substitution,
    ) -> ControlFlow<VisiblyUninhabited> {
        let variant_data = self.db.variant_fields(variant);
        let fields = variant_data.fields();
        if fields.is_empty() {
            return CONTINUE_OPAQUELY_INHABITED;
        }

        let is_enum = matches!(variant, VariantId::EnumVariantId(..));
        let field_tys = self.db.field_types(variant);
        let field_vis = if is_enum { None } else { Some(self.db.field_visibilities(variant)) };

        for (fid, _) in fields.iter() {
            self.visit_field(field_vis.as_ref().map(|it| it[fid]), &field_tys[fid], subst)?;
        }
        CONTINUE_OPAQUELY_INHABITED
    }

    fn visit_field(
        &mut self,
        vis: Option<Visibility>,
        ty: &Binders<Ty>,
        subst: &Substitution,
    ) -> ControlFlow<VisiblyUninhabited> {
        if vis.is_none_or(|it| it.is_visible_from(self.db, self.target_mod)) {
            let ty = ty.clone().substitute(Interner, subst);
            ty.visit_with(self, DebruijnIndex::INNERMOST)
        } else {
            CONTINUE_OPAQUELY_INHABITED
        }
    }
}
