//! Type inhabitedness logic.
use std::ops::ControlFlow::{self, Break, Continue};

use chalk_ir::{
    visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
    DebruijnIndex,
};
use hir_def::{
    adt::VariantData, attr::Attrs, visibility::Visibility, AdtId, EnumVariantId, HasModule, Lookup,
    ModuleId, VariantId,
};

use crate::{
    consteval::try_const_usize, db::HirDatabase, Binders, Interner, Substitution, Ty, TyKind,
};

/// Checks whether a type is visibly uninhabited from a particular module.
pub(crate) fn is_ty_uninhabited_from(ty: &Ty, target_mod: ModuleId, db: &dyn HirDatabase) -> bool {
    let mut uninhabited_from = UninhabitedFrom { target_mod, db };
    let inhabitedness = ty.visit_with(&mut uninhabited_from, DebruijnIndex::INNERMOST);
    inhabitedness == BREAK_VISIBLY_UNINHABITED
}

/// Checks whether a variant is visibly uninhabited from a particular module.
pub(crate) fn is_enum_variant_uninhabited_from(
    variant: EnumVariantId,
    subst: &Substitution,
    target_mod: ModuleId,
    db: &dyn HirDatabase,
) -> bool {
    let enum_data = db.enum_data(variant.parent);
    let vars_attrs = db.variants_attrs(variant.parent);
    let is_local = variant.parent.lookup(db.upcast()).container.krate() == target_mod.krate();

    let mut uninhabited_from = UninhabitedFrom { target_mod, db };
    let inhabitedness = uninhabited_from.visit_variant(
        variant.into(),
        &enum_data.variants[variant.local_id].variant_data,
        subst,
        &vars_attrs[variant.local_id],
        is_local,
    );
    inhabitedness == BREAK_VISIBLY_UNINHABITED
}

struct UninhabitedFrom<'a> {
    target_mod: ModuleId,
    db: &'a dyn HirDatabase,
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
        match ty.kind(Interner) {
            TyKind::Adt(adt, subst) => self.visit_adt(adt.0, subst),
            TyKind::Never => BREAK_VISIBLY_UNINHABITED,
            TyKind::Tuple(..) => ty.super_visit_with(self, outer_binder),
            TyKind::Array(item_ty, len) => match try_const_usize(len) {
                Some(0) | None => CONTINUE_OPAQUELY_INHABITED,
                Some(1..) => item_ty.super_visit_with(self, outer_binder),
            },

            TyKind::Ref(..) | _ => CONTINUE_OPAQUELY_INHABITED,
        }
    }

    fn interner(&self) -> Interner {
        Interner
    }
}

impl UninhabitedFrom<'_> {
    fn visit_adt(&mut self, adt: AdtId, subst: &Substitution) -> ControlFlow<VisiblyUninhabited> {
        let attrs = self.db.attrs(adt.into());
        let adt_non_exhaustive = attrs.by_key("non_exhaustive").exists();
        let is_local = adt.module(self.db.upcast()).krate() == self.target_mod.krate();
        if adt_non_exhaustive && !is_local {
            return CONTINUE_OPAQUELY_INHABITED;
        }

        // An ADT is uninhabited iff all its variants uninhabited.
        match adt {
            // rustc: For now, `union`s are never considered uninhabited.
            AdtId::UnionId(_) => CONTINUE_OPAQUELY_INHABITED,
            AdtId::StructId(s) => {
                let struct_data = self.db.struct_data(s);
                self.visit_variant(s.into(), &struct_data.variant_data, subst, &attrs, is_local)
            }
            AdtId::EnumId(e) => {
                let vars_attrs = self.db.variants_attrs(e);
                let enum_data = self.db.enum_data(e);

                for (local_id, enum_var) in enum_data.variants.iter() {
                    let variant_inhabitedness = self.visit_variant(
                        EnumVariantId { parent: e, local_id }.into(),
                        &enum_var.variant_data,
                        subst,
                        &vars_attrs[local_id],
                        is_local,
                    );
                    match variant_inhabitedness {
                        Break(VisiblyUninhabited) => continue,
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
        variant_data: &VariantData,
        subst: &Substitution,
        attrs: &Attrs,
        is_local: bool,
    ) -> ControlFlow<VisiblyUninhabited> {
        let non_exhaustive_field_list = attrs.by_key("non_exhaustive").exists();
        if non_exhaustive_field_list && !is_local {
            return CONTINUE_OPAQUELY_INHABITED;
        }

        let is_enum = matches!(variant, VariantId::EnumVariantId(..));
        let field_tys = self.db.field_types(variant);
        let field_vis = self.db.field_visibilities(variant);

        for (fid, _) in variant_data.fields().iter() {
            self.visit_field(field_vis[fid], &field_tys[fid], subst, is_enum)?;
        }
        CONTINUE_OPAQUELY_INHABITED
    }

    fn visit_field(
        &mut self,
        vis: Visibility,
        ty: &Binders<Ty>,
        subst: &Substitution,
        is_enum: bool,
    ) -> ControlFlow<VisiblyUninhabited> {
        if is_enum || vis.is_visible_from(self.db.upcast(), self.target_mod) {
            let ty = ty.clone().substitute(Interner, subst);
            ty.visit_with(self, DebruijnIndex::INNERMOST)
        } else {
            CONTINUE_OPAQUELY_INHABITED
        }
    }
}
