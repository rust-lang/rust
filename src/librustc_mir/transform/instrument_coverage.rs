use crate::transform::{MirPass, MirSource};
use crate::util::patch::MirPatch;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::lang_items;
use rustc_hir::*;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::{
    self, BasicBlock, BasicBlockData, CoverageData, Operand, Place, SourceInfo, StatementKind,
    Terminator, TerminatorKind, START_BLOCK,
};
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::Span;

/// Inserts call to count_code_region() as a placeholder to be replaced during code generation with
/// the intrinsic llvm.instrprof.increment.
pub struct InstrumentCoverage;

struct Instrumentor<'tcx> {
    tcx: TyCtxt<'tcx>,
    num_counters: u32,
}

impl<'tcx> MirPass<'tcx> for InstrumentCoverage {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, mir_body: &mut mir::Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.instrument_coverage {
            // If the InstrumentCoverage pass is called on promoted MIRs, skip them.
            // See: https://github.com/rust-lang/rust/pull/73011#discussion_r438317601
            if src.promoted.is_none() {
                assert!(mir_body.coverage_data.is_none());

                let hash = hash_mir_source(tcx, &src);

                debug!(
                    "instrumenting {:?}, hash: {}, span: {}",
                    src.def_id(),
                    hash,
                    tcx.sess.source_map().span_to_string(mir_body.span)
                );

                let num_counters = Instrumentor::new(tcx).inject_counters(mir_body);

                mir_body.coverage_data = Some(CoverageData { hash, num_counters });
            }
        }
    }
}

impl<'tcx> Instrumentor<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, num_counters: 0 }
    }

    fn next_counter(&mut self) -> u32 {
        let next = self.num_counters;
        self.num_counters += 1;
        next
    }

    fn inject_counters(&mut self, mir_body: &mut mir::Body<'tcx>) -> u32 {
        // FIXME(richkadel): As a first step, counters are only injected at the top of each
        // function. The complete solution will inject counters at each conditional code branch.
        let top_of_function = START_BLOCK;
        let entire_function = mir_body.span;

        self.inject_counter(mir_body, top_of_function, entire_function);

        self.num_counters
    }

    fn inject_counter(
        &mut self,
        mir_body: &mut mir::Body<'tcx>,
        next_block: BasicBlock,
        code_region: Span,
    ) {
        let injection_point = code_region.shrink_to_lo();

        let count_code_region_fn = function_handle(
            self.tcx,
            self.tcx.require_lang_item(lang_items::CountCodeRegionFnLangItem, None),
            injection_point,
        );
        let counter_index = Operand::const_from_scalar(
            self.tcx,
            self.tcx.types.u32,
            Scalar::from_u32(self.next_counter()),
            injection_point,
        );

        let mut patch = MirPatch::new(mir_body);

        let temp = patch.new_temp(self.tcx.mk_unit(), code_region);
        let new_block = patch.new_block(placeholder_block(code_region));
        patch.patch_terminator(
            new_block,
            TerminatorKind::Call {
                func: count_code_region_fn,
                args: vec![counter_index],
                // new_block will swapped with the next_block, after applying patch
                destination: Some((Place::from(temp), new_block)),
                cleanup: None,
                from_hir_call: false,
                fn_span: injection_point,
            },
        );

        patch.add_statement(new_block.start_location(), StatementKind::StorageLive(temp));
        patch.add_statement(next_block.start_location(), StatementKind::StorageDead(temp));

        patch.apply(mir_body);

        // To insert the `new_block` in front of the first block in the counted branch (the
        // `next_block`), just swap the indexes, leaving the rest of the graph unchanged.
        mir_body.basic_blocks_mut().swap(next_block, new_block);
    }
}

fn function_handle<'tcx>(tcx: TyCtxt<'tcx>, fn_def_id: DefId, span: Span) -> Operand<'tcx> {
    let ret_ty = tcx.fn_sig(fn_def_id).output();
    let ret_ty = ret_ty.no_bound_vars().unwrap();
    let substs = tcx.mk_substs(::std::iter::once(ty::subst::GenericArg::from(ret_ty)));
    Operand::function_handle(tcx, fn_def_id, substs, span)
}

fn placeholder_block(span: Span) -> BasicBlockData<'tcx> {
    BasicBlockData {
        statements: vec![],
        terminator: Some(Terminator {
            source_info: SourceInfo::outermost(span),
            // this gets overwritten by the counter Call
            kind: TerminatorKind::Unreachable,
        }),
        is_cleanup: false,
    }
}

fn hash_mir_source<'tcx>(tcx: TyCtxt<'tcx>, src: &MirSource<'tcx>) -> u64 {
    let fn_body_id = match tcx.hir().get_if_local(src.def_id()) {
        Some(node) => match associated_body(node) {
            Some(body_id) => body_id,
            _ => bug!("instrumented MirSource does not include a function body: {:?}", node),
        },
        None => bug!("instrumented MirSource is not local: {:?}", src),
    };
    let hir_body = tcx.hir().body(fn_body_id);
    let mut hcx = tcx.create_no_span_stable_hashing_context();
    hash(&mut hcx, &hir_body.value).to_smaller_hash()
}

fn hash(
    hcx: &mut StableHashingContext<'tcx>,
    node: &impl HashStable<StableHashingContext<'tcx>>,
) -> Fingerprint {
    let mut stable_hasher = StableHasher::new();
    node.hash_stable(hcx, &mut stable_hasher);
    stable_hasher.finish()
}

fn associated_body<'hir>(node: Node<'hir>) -> Option<BodyId> {
    match node {
        Node::Item(Item {
            kind: ItemKind::Const(_, body) | ItemKind::Static(.., body) | ItemKind::Fn(.., body),
            ..
        })
        | Node::TraitItem(TraitItem {
            kind:
                TraitItemKind::Const(_, Some(body)) | TraitItemKind::Fn(_, TraitFn::Provided(body)),
            ..
        })
        | Node::ImplItem(ImplItem {
            kind: ImplItemKind::Const(_, body) | ImplItemKind::Fn(_, body),
            ..
        })
        | Node::Expr(Expr { kind: ExprKind::Closure(.., body, _, _), .. }) => Some(*body),

        Node::AnonConst(constant) => Some(constant.body),

        _ => None,
    }
}
