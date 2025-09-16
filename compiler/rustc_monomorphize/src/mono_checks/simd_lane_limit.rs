use rustc_abi::FieldIdx;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor};
use rustc_span::{Span, sym};

use crate::errors;

fn get_simd_lane_limit<'tcx>(tcx: TyCtxt<'tcx>, def_id: rustc_hir::def_id::DefId) -> Option<u64> {
    tcx.get_attrs_by_path(def_id, &[sym::rustc_simd_monomorphize_lane_limit])
        .next()
        .and_then(|attr| attr.value_lit())
        .and_then(|lit| lit.symbol.as_str().parse::<u64>().ok())
}

fn get_simd_lane_count<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::AdtDef<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
) -> Option<u64> {
    let fields = &def.non_enum_variant().fields;
    let field_ty = fields[FieldIdx::ZERO].ty(tcx, args);
    match field_ty.kind() {
        ty::Array(_, len) => match len.try_to_target_usize(tcx) {
            Some(n) => Some(n),
            None => {
                tcx.dcx().delayed_bug("unable to evaluate SIMD array length");
                None
            }
        },
        _ => {
            tcx.dcx().delayed_bug("expected SIMD vector to be an array type");
            None
        }
    }
}

struct SimdLaneVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    current_span: Span,
    seen: FxHashSet<Ty<'tcx>>,
}

impl<'tcx> SimdLaneVisitor<'tcx> {
    fn with_span<F: FnOnce(&mut Self)>(&mut self, span: Span, f: F) {
        let prev = self.current_span;
        self.current_span = span;
        f(self);
        self.current_span = prev;
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for SimdLaneVisitor<'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if !self.seen.insert(ty) {
            return;
        }

        if let ty::Adt(def, args) = ty.kind() {
            // Enforce SIMD lane limit when present
            if def.repr().simd() {
                if let Some(limit) = get_simd_lane_limit(self.tcx, def.did()) {
                    if let Some(lanes) = get_simd_lane_count(self.tcx, *def, args) {
                        if lanes > limit {
                            self.tcx.dcx().emit_err(errors::SimdMonoLaneLimitExceeded {
                                span: self.current_span,
                                ty,
                                lanes,
                                limit,
                            });
                        }
                    }
                }
            }

            // Recurse into field types
            for variant in def.variants() {
                for field in &variant.fields {
                    let field_ty = field.ty(self.tcx, args);
                    field_ty.visit_with(self);
                }
            }
        }

        ty.super_visit_with(self);
    }
}

pub(crate) fn check_simd_lane_limits<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    body: &'tcx mir::Body<'tcx>,
) {
    // Check function signatures
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let mut visitor =
        SimdLaneVisitor { tcx, current_span: Span::default(), seen: FxHashSet::default() };

    if let Ok(abi) =
        tcx.fn_abi_of_instance(typing_env.as_query_input((instance, ty::List::empty())))
    {
        let def_span = tcx.def_span(instance.def_id());
        for arg in abi.args.iter() {
            visitor.with_span(def_span, |v| arg.layout.ty.visit_with(v));
        }
        visitor.with_span(def_span, |v| abi.ret.layout.ty.visit_with(v));
    }

    // Check locals
    for local in body.local_decls.iter() {
        let ty = local.ty;
        let ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(ty),
        );
        let span = local.source_info.span;
        visitor.with_span(span, |v| ty.visit_with(v));
    }
}
