use crate::hir::def::{Res, DefKind};
use crate::hir::def_id::DefId;
use crate::ty::{self, Ty, TyCtxt};
use crate::ty::layout::{LayoutError, Pointer, SizeSkeleton, VariantIdx};
use crate::ty::query::Providers;

use rustc_target::spec::abi::Abi::RustIntrinsic;
use rustc_data_structures::indexed_vec::Idx;
use syntax_pos::{Span, sym};
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::hir;

fn check_mod_intrinsics(tcx: TyCtxt<'_>, module_def_id: DefId) {
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut ItemVisitor { tcx }.as_deep_visitor()
    );
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        check_mod_intrinsics,
        ..*providers
    };
}

struct ItemVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

struct ExprVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    tables: &'tcx ty::TypeckTables<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

/// If the type is `Option<T>`, it will return `T`, otherwise
/// the type itself. Works on most `Option`-like types.
fn unpack_option_like<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    let (def, substs) = match ty.sty {
        ty::Adt(def, substs) => (def, substs),
        _ => return ty
    };

    if def.variants.len() == 2 && !def.repr.c() && def.repr.int.is_none() {
        let data_idx;

        let one = VariantIdx::new(1);
        let zero = VariantIdx::new(0);

        if def.variants[zero].fields.is_empty() {
            data_idx = one;
        } else if def.variants[one].fields.is_empty() {
            data_idx = zero;
        } else {
            return ty;
        }

        if def.variants[data_idx].fields.len() == 1 {
            return def.variants[data_idx].fields[0].ty(tcx, substs);
        }
    }

    ty
}

impl ExprVisitor<'tcx> {
    fn def_id_is_transmute(&self, def_id: DefId) -> bool {
        self.tcx.fn_sig(def_id).abi() == RustIntrinsic &&
        self.tcx.item_name(def_id) == sym::transmute
    }

    fn check_transmute(&self, span: Span, from: Ty<'tcx>, to: Ty<'tcx>) {
        let sk_from = SizeSkeleton::compute(from, self.tcx, self.param_env);
        let sk_to = SizeSkeleton::compute(to, self.tcx, self.param_env);

        // Check for same size using the skeletons.
        if let (Ok(sk_from), Ok(sk_to)) = (sk_from, sk_to) {
            if sk_from.same_size(sk_to) {
                return;
            }

            // Special-case transmutting from `typeof(function)` and
            // `Option<typeof(function)>` to present a clearer error.
            let from = unpack_option_like(self.tcx.global_tcx(), from);
            if let (&ty::FnDef(..), SizeSkeleton::Known(size_to)) = (&from.sty, sk_to) {
                if size_to == Pointer.size(&self.tcx) {
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
                    format!("pointer to `{}`", tail)
                }
                Err(LayoutError::Unknown(bad)) => {
                    if bad == ty {
                        "this type does not have a fixed size".to_owned()
                    } else {
                        format!("size can vary because of {}", bad)
                    }
                }
                Err(err) => err.to_string()
            }
        };

        let mut err = struct_span_err!(self.tcx.sess, span, E0512,
                                       "cannot transmute between types of different sizes, \
                                        or dependently-sized types");
        if from == to {
            err.note(&format!("`{}` does not have a fixed size", from));
        } else {
            err.note(&format!("source type: `{}` ({})", from, skeleton_string(from, sk_from)))
                .note(&format!("target type: `{}` ({})", to, skeleton_string(to, sk_to)));
        }
        err.emit()
    }
}

impl Visitor<'tcx> for ItemVisitor<'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let owner_def_id = self.tcx.hir().body_owner_def_id(body_id);
        let body = self.tcx.hir().body(body_id);
        let param_env = self.tcx.param_env(owner_def_id);
        let tables = self.tcx.typeck_tables_of(owner_def_id);
        ExprVisitor { tcx: self.tcx, param_env, tables }.visit_body(body);
        self.visit_body(body);
    }
}

impl Visitor<'tcx> for ExprVisitor<'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        let res = if let hir::ExprKind::Path(ref qpath) = expr.node {
            self.tables.qpath_res(qpath, expr.hir_id)
        } else {
            Res::Err
        };
        if let Res::Def(DefKind::Fn, did) = res {
            if self.def_id_is_transmute(did) {
                let typ = self.tables.node_type(expr.hir_id);
                let sig = typ.fn_sig(self.tcx);
                let from = sig.inputs().skip_binder()[0];
                let to = *sig.output().skip_binder();
                self.check_transmute(expr.span, from, to);
            }
        }

        intravisit::walk_expr(self, expr);
    }
}
