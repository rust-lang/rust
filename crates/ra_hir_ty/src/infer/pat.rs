//! Type inference for patterns.

use std::iter::repeat;
use std::sync::Arc;

use hir_def::{
    expr::{BindingAnnotation, Pat, PatId, RecordFieldPat},
    path::Path,
    type_ref::Mutability,
};
use hir_expand::name::Name;
use test_utils::tested_by;

use super::{BindingMode, InferenceContext};
use crate::{db::HirDatabase, utils::variant_data, Substs, Ty, TypeCtor, TypeWalk};

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    fn infer_tuple_struct_pat(
        &mut self,
        path: Option<&Path>,
        subpats: &[PatId],
        expected: &Ty,
        default_bm: BindingMode,
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path);
        let var_data = def.map(|it| variant_data(self.db, it));
        self.unify(&ty, expected);

        let substs = ty.substs().unwrap_or_else(Substs::empty);

        let field_tys = def.map(|it| self.db.field_types(it.into())).unwrap_or_default();

        for (i, &subpat) in subpats.iter().enumerate() {
            let expected_ty = var_data
                .as_ref()
                .and_then(|d| d.field(&Name::new_tuple_field(i)))
                .map_or(Ty::Unknown, |field| field_tys[field].clone())
                .subst(&substs);
            let expected_ty = self.normalize_associated_types_in(expected_ty);
            self.infer_pat(subpat, &expected_ty, default_bm);
        }

        ty
    }

    fn infer_record_pat(
        &mut self,
        path: Option<&Path>,
        subpats: &[RecordFieldPat],
        expected: &Ty,
        default_bm: BindingMode,
        id: PatId,
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path);
        let var_data = def.map(|it| variant_data(self.db, it));
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }

        self.unify(&ty, expected);

        let substs = ty.substs().unwrap_or_else(Substs::empty);

        let field_tys = def.map(|it| self.db.field_types(it.into())).unwrap_or_default();
        for subpat in subpats {
            let matching_field = var_data.as_ref().and_then(|it| it.field(&subpat.name));
            let expected_ty =
                matching_field.map_or(Ty::Unknown, |field| field_tys[field].clone()).subst(&substs);
            let expected_ty = self.normalize_associated_types_in(expected_ty);
            self.infer_pat(subpat.pat, &expected_ty, default_bm);
        }

        ty
    }

    pub(super) fn infer_pat(
        &mut self,
        pat: PatId,
        mut expected: &Ty,
        mut default_bm: BindingMode,
    ) -> Ty {
        let body = Arc::clone(&self.body); // avoid borrow checker problem

        let is_non_ref_pat = match &body[pat] {
            Pat::Tuple(..)
            | Pat::TupleStruct { .. }
            | Pat::Record { .. }
            | Pat::Range { .. }
            | Pat::Slice { .. } => true,
            // FIXME: Path/Lit might actually evaluate to ref, but inference is unimplemented.
            Pat::Path(..) | Pat::Lit(..) => true,
            Pat::Wild | Pat::Bind { .. } | Pat::Ref { .. } | Pat::Missing => false,
        };
        if is_non_ref_pat {
            while let Some((inner, mutability)) = expected.as_reference() {
                expected = inner;
                default_bm = match default_bm {
                    BindingMode::Move => BindingMode::Ref(mutability),
                    BindingMode::Ref(Mutability::Shared) => BindingMode::Ref(Mutability::Shared),
                    BindingMode::Ref(Mutability::Mut) => BindingMode::Ref(mutability),
                }
            }
        } else if let Pat::Ref { .. } = &body[pat] {
            tested_by!(match_ergonomics_ref);
            // When you encounter a `&pat` pattern, reset to Move.
            // This is so that `w` is by value: `let (_, &w) = &(1, &2);`
            default_bm = BindingMode::Move;
        }

        // Lose mutability.
        let default_bm = default_bm;
        let expected = expected;

        let ty = match &body[pat] {
            Pat::Tuple(ref args) => {
                let expectations = match expected.as_tuple() {
                    Some(parameters) => &*parameters.0,
                    _ => &[],
                };
                let expectations_iter = expectations.iter().chain(repeat(&Ty::Unknown));

                let inner_tys = args
                    .iter()
                    .zip(expectations_iter)
                    .map(|(&pat, ty)| self.infer_pat(pat, ty, default_bm))
                    .collect();

                Ty::apply(TypeCtor::Tuple { cardinality: args.len() as u16 }, Substs(inner_tys))
            }
            Pat::Ref { pat, mutability } => {
                let expectation = match expected.as_reference() {
                    Some((inner_ty, exp_mut)) => {
                        if *mutability != exp_mut {
                            // FIXME: emit type error?
                        }
                        inner_ty
                    }
                    _ => &Ty::Unknown,
                };
                let subty = self.infer_pat(*pat, expectation, default_bm);
                Ty::apply_one(TypeCtor::Ref(*mutability), subty)
            }
            Pat::TupleStruct { path: p, args: subpats } => {
                self.infer_tuple_struct_pat(p.as_ref(), subpats, expected, default_bm)
            }
            Pat::Record { path: p, args: fields } => {
                self.infer_record_pat(p.as_ref(), fields, expected, default_bm, pat)
            }
            Pat::Path(path) => {
                // FIXME use correct resolver for the surrounding expression
                let resolver = self.resolver.clone();
                self.infer_path(&resolver, &path, pat.into()).unwrap_or(Ty::Unknown)
            }
            Pat::Bind { mode, name: _, subpat } => {
                let mode = if mode == &BindingAnnotation::Unannotated {
                    default_bm
                } else {
                    BindingMode::convert(*mode)
                };
                let inner_ty = if let Some(subpat) = subpat {
                    self.infer_pat(*subpat, expected, default_bm)
                } else {
                    expected.clone()
                };
                let inner_ty = self.insert_type_vars_shallow(inner_ty);

                let bound_ty = match mode {
                    BindingMode::Ref(mutability) => {
                        Ty::apply_one(TypeCtor::Ref(mutability), inner_ty.clone())
                    }
                    BindingMode::Move => inner_ty.clone(),
                };
                let bound_ty = self.resolve_ty_as_possible(bound_ty);
                self.write_pat_ty(pat, bound_ty);
                return inner_ty;
            }
            _ => Ty::Unknown,
        };
        // use a new type variable if we got Ty::Unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.unify(&ty, expected);
        let ty = self.resolve_ty_as_possible(ty);
        self.write_pat_ty(pat, ty.clone());
        ty
    }
}
