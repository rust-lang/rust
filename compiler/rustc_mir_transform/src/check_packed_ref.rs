use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::lint::builtin::UNALIGNED_REFERENCES;

use crate::util;
use crate::MirLint;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { unsafe_derive_on_repr_packed, ..*providers };
}

pub struct CheckPackedRef;

impl<'tcx> MirLint<'tcx> for CheckPackedRef {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let param_env = tcx.param_env(body.source.def_id());
        let source_info = SourceInfo::outermost(body.span);
        let mut checker = PackedRefChecker { body, tcx, param_env, source_info };
        checker.visit_body(&body);
    }
}

struct PackedRefChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    source_info: SourceInfo,
}

fn unsafe_derive_on_repr_packed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let lint_hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    // FIXME: when we make this a hard error, this should have its
    // own error code.

    let extra = if tcx.generics_of(def_id).own_requires_monomorphization() {
        "with type or const parameters"
    } else {
        "that does not derive `Copy`"
    };
    let message = format!(
        "`{}` can't be derived on this `#[repr(packed)]` struct {}",
        tcx.item_name(tcx.trait_id_of_impl(def_id.to_def_id()).expect("derived trait name")),
        extra
    );

    tcx.struct_span_lint_hir(
        UNALIGNED_REFERENCES,
        lint_hir_id,
        tcx.def_span(def_id),
        message,
        |lint| lint,
    );
}

impl<'tcx> Visitor<'tcx> for PackedRefChecker<'_, 'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // Make sure we know where in the MIR we are.
        self.source_info = terminator.source_info;
        self.super_terminator(terminator, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // Make sure we know where in the MIR we are.
        self.source_info = statement.source_info;
        self.super_statement(statement, location);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        if context.is_borrow() {
            if util::is_disaligned(self.tcx, self.body, self.param_env, *place) {
                let def_id = self.body.source.instance.def_id();
                if let Some(impl_def_id) = self
                    .tcx
                    .impl_of_method(def_id)
                    .filter(|&def_id| self.tcx.is_builtin_derive(def_id))
                {
                    // If a method is defined in the local crate,
                    // the impl containing that method should also be.
                    self.tcx.ensure().unsafe_derive_on_repr_packed(impl_def_id.expect_local());
                } else {
                    let source_info = self.source_info;
                    let lint_root = self.body.source_scopes[source_info.scope]
                        .local_data
                        .as_ref()
                        .assert_crate_local()
                        .lint_root;
                    self.tcx.struct_span_lint_hir(
                        UNALIGNED_REFERENCES,
                        lint_root,
                        source_info.span,
                        "reference to packed field is unaligned",
                        |lint| {
                            lint
                                .note(
                                    "fields of packed structs are not properly aligned, and creating \
                                    a misaligned reference is undefined behavior (even if that \
                                    reference is never dereferenced)",
                                )
                                .help(
                                    "copy the field contents to a local variable, or replace the \
                                    reference with a raw pointer and use `read_unaligned`/`write_unaligned` \
                                    (loads and stores via `*p` must be properly aligned even when using raw pointers)"
                                )
                        },
                    );
                }
            }
        }
    }
}
