// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir;
use hir::def::Def;
use hir::def_id::DefId;
use hir::intravisit::{self, Visitor, NestedVisitorMap};
use ty::layout::{LayoutError, Pointer, SizeSkeleton};
use ty::{self, Ty, AdtKind, TyCtxt, TypeFoldable};
use ty::subst::Substs;

use rustc_target::spec::abi::Abi::RustIntrinsic;
use syntax_pos::DUMMY_SP;
use syntax_pos::Span;

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut visitor = ItemVisitor {
        tcx,
    };
    tcx.hir.krate().visit_all_item_likes(&mut visitor.as_deep_visitor());
}

struct ItemVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

struct ExprVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'tcx ty::TypeckTables<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

/// If the type is `Option<T>`, it will return `T`, otherwise
/// the type itself. Works on most `Option`-like types.
fn unpack_option_like<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                ty: Ty<'tcx>)
                                -> Ty<'tcx> {
    let (def, substs) = match ty.sty {
        ty::TyAdt(def, substs) => (def, substs),
        _ => return ty
    };

    if def.variants.len() == 2 && !def.repr.c() && def.repr.int.is_none() {
        let data_idx;

        if def.variants[0].fields.is_empty() {
            data_idx = 1;
        } else if def.variants[1].fields.is_empty() {
            data_idx = 0;
        } else {
            return ty;
        }

        if def.variants[data_idx].fields.len() == 1 {
            return def.variants[data_idx].fields[0].ty(tcx, substs);
        }
    }

    ty
}

/// Check if this enum can be safely exported based on the
/// "nullable pointer optimization". Currently restricted
/// to function pointers and references, but could be
/// expanded to cover NonZero raw pointers and newtypes.
/// FIXME: This duplicates code in codegen.
pub fn is_repr_nullable_ptr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  def: &'tcx ty::AdtDef,
                                  substs: &Substs<'tcx>)
                                  -> bool {
    if def.variants.len() == 2 {
        let data_idx;

        if def.variants[0].fields.is_empty() {
            data_idx = 1;
        } else if def.variants[1].fields.is_empty() {
            data_idx = 0;
        } else {
            return false;
        }

        if def.variants[data_idx].fields.len() == 1 {
            match def.variants[data_idx].fields[0].ty(tcx, substs).sty {
                ty::TyFnPtr(_) => {
                    return true;
                }
                ty::TyRef(..) => {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

impl<'a, 'tcx> ExprVisitor<'a, 'tcx> {
    fn def_id_is_transmute(&self, def_id: DefId) -> bool {
        self.tcx.fn_sig(def_id).abi() == RustIntrinsic &&
        self.tcx.item_name(def_id) == "transmute"
    }

    /// Calculate whether a type has a specified layout.
    ///
    /// The function returns `None` in indeterminate cases (such as `TyError`).
    fn is_layout_specified(&self, ty: Ty<'tcx>) -> Option<bool> {
        match ty.sty {
            // These types have a specified layout
            // Reference: Primitive type layout
            ty::TyBool |
            ty::TyChar |
            ty::TyInt(_) |
            ty::TyUint(_) |
            ty::TyFloat(_) |
            // Reference: Pointers and references layout
            ty::TyFnPtr(_) => Some(true),
            // Reference: Array layout (depends on the contained type)
            ty::TyArray(ty, _) => self.is_layout_specified(ty),
            // Reference: Tuple layout (only specified if empty)
            ty::TyTuple(ref tys) => Some(tys.is_empty()),

            // Cases with definitely unspecified layouts
            ty::TyClosure(_, _) |
            ty::TyGenerator(_, _, _) |
            ty::TyGeneratorWitness(_) => Some(false),
            // Currently ZST, but does not seem to be guaranteed
            ty::TyFnDef(_, _) => Some(false),
            // Unsized types
            ty::TyForeign(_) |
            ty::TyNever |
            ty::TyStr |
            ty::TySlice(_) |
            ty::TyDynamic(_, _) => Some(false),

            // Indeterminate cases
            ty::TyInfer(_) |
            // should we report `Some(false)` for TyParam(_)? It this possible to reach this branch?
            ty::TyParam(_) |
            ty::TyError => None,

            // The “it’s complicated™” cases
            // Reference: Pointers and references layout
            ty::TyRawPtr(ty::TypeAndMut { ty: pointee, .. }) |
            ty::TyRef(_, pointee, _) => {
                let pointee = self.tcx.normalize_erasing_regions(self.param_env, pointee);
                // Pointers to unsized types have no specified layout.
                Some(pointee.is_sized(self.tcx.at(DUMMY_SP), self.param_env))
            }
            ty::TyProjection(_) | ty::TyAnon(_, _) => {
                let normalized = self.tcx.normalize_erasing_regions(self.param_env, ty);
                if ty == normalized {
                    None
                } else {
                    self.is_layout_specified(normalized)
                }
            }
            ty::TyAdt(def, substs) => {
                // Documentation guarantees 0-size.
                if def.is_phantom_data() {
                    return Some(true);
                }
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        if !def.repr.c() && !def.repr.transparent() && !def.repr.simd() {
                            return Some(false);
                        }
                        // FIXME: do we guarantee 0-sizedness for structs with 0 fields?
                        // If not, they should cause Some(false) here.
                        let mut seen_none = false;
                        for field in &def.non_enum_variant().fields {
                            let field_ty = field.ty(self.tcx, substs);
                            match self.is_layout_specified(field_ty) {
                                Some(true) => continue,
                                None => {
                                    seen_none = true;
                                    continue;
                                }
                                x => return x,
                            }
                        }
                        return if seen_none { None } else { Some(true) };
                    }
                    AdtKind::Enum => {
                        if !def.repr.c() && def.repr.int.is_none() {
                            if !is_repr_nullable_ptr(self.tcx, def, substs) {
                                return Some(false);
                            }
                        }
                        return Some(true);
                    }
                }
            }
        }
    }

    fn check_transmute(&self, span: Span, from: Ty<'tcx>, to: Ty<'tcx>) {
        // Check for unspecified types before checking for same size.
        assert!(!from.has_infer_types());
        assert!(!to.has_infer_types());

        let unspecified_layout = |msg, ty| {
            if ::std::env::var("RUSTC_BOOTSTRAP").is_ok() {
                struct_span_warn!(self.tcx.sess, span, E0912, "{}", msg)
                    .note(&format!("{} has an unspecified layout", ty))
                    .note("this will become a hard error in the future")
                    .emit();
            } else {
                struct_span_err!(self.tcx.sess, span, E0912, "{}", msg)
                    .note(&format!("{} has an unspecified layout", ty))
                    .note("this will become a hard error in the future")
                    .emit();
            }
        };

        if self.is_layout_specified(from) == Some(false) {
            unspecified_layout("transmutation from a type with an unspecified layout", from);
        }

        if self.is_layout_specified(to) == Some(false) {
            unspecified_layout("transmutation to a type with an unspecified layout", to);
        }

        // Check for same size using the skeletons.
        let sk_from = SizeSkeleton::compute(from, self.tcx, self.param_env);
        let sk_to = SizeSkeleton::compute(to, self.tcx, self.param_env);

        if let (Ok(sk_from), Ok(sk_to)) = (sk_from, sk_to) {
            if sk_from.same_size(sk_to) {
                return;
            }

            // Special-case transmutting from `typeof(function)` and
            // `Option<typeof(function)>` to present a clearer error.
            let from = unpack_option_like(self.tcx.global_tcx(), from);
            if let (&ty::TyFnDef(..), SizeSkeleton::Known(size_to)) = (&from.sty, sk_to) {
                if size_to == Pointer.size(self.tcx) {
                    struct_span_err!(self.tcx.sess, span, E0591,
                                     "can't transmute zero-sized type")
                        .note(&format!("source type: {}", from))
                        .note(&format!("target type: {}", to))
                        .help("cast with `as` to a pointer instead")
                        .emit();
                    return;
                }
            }
        }

        // Try to display a sensible error with as much information as possible.
        let skeleton_string = |ty: Ty<'tcx>, sk| {
            match sk {
                Ok(SizeSkeleton::Known(size)) => {
                    format!("{} bits", size.bits())
                }
                Ok(SizeSkeleton::Pointer { tail, .. }) => {
                    format!("pointer to {}", tail)
                }
                Err(LayoutError::Unknown(bad)) => {
                    if bad == ty {
                        format!("this type's size can vary")
                    } else {
                        format!("size can vary because of {}", bad)
                    }
                }
                Err(err) => err.to_string()
            }
        };

        struct_span_err!(self.tcx.sess, span, E0512,
            "transmute called with types of different sizes")
            .note(&format!("source type: {} ({})", from, skeleton_string(from, sk_from)))
            .note(&format!("target type: {} ({})", to, skeleton_string(to, sk_to)))
            .emit();
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ItemVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let owner_def_id = self.tcx.hir.body_owner_def_id(body_id);
        let body = self.tcx.hir.body(body_id);
        let param_env = self.tcx.param_env(owner_def_id);
        let tables = self.tcx.typeck_tables_of(owner_def_id);
        ExprVisitor { tcx: self.tcx, param_env, tables }.visit_body(body);
        self.visit_body(body);
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ExprVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        let def = if let hir::ExprPath(ref qpath) = expr.node {
            self.tables.qpath_def(qpath, expr.hir_id)
        } else {
            Def::Err
        };
        if let Def::Fn(did) = def {
            if self.def_id_is_transmute(did) {
                let typ = self.tables.node_id_to_type(expr.hir_id);
                let sig = typ.fn_sig(self.tcx);
                let from = sig.inputs().skip_binder()[0];
                let to = *sig.output().skip_binder();
                self.check_transmute(expr.span, from, to);
            }
        }

        intravisit::walk_expr(self, expr);
    }
}
