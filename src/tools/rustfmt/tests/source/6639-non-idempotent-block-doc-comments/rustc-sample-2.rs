//! Taken from
//! <https://github.com/rust-lang/rust/blob/70222712809cd5cc1718ed8995914a1cbacb6b92/compiler/rustc_mir_transform/src/inline.rs#L1166-L1186>.

/**
 * Integrator.
 *
 * Integrates blocks from the callee function into the calling function.
 * Updates block indices, references to locals and other control flow
 * stuff.
*/
struct Integrator<'a, 'tcx> {
    args: &'a [Local],
    new_locals: RangeFrom<Local>,
    new_scopes: RangeFrom<SourceScope>,
    new_blocks: RangeFrom<BasicBlock>,
    destination: Local,
    callsite_scope: SourceScopeData<'tcx>,
    callsite: &'a CallSite<'tcx>,
    cleanup_block: UnwindAction,
    in_cleanup_block: bool,
    return_block: Option<BasicBlock>,
    tcx: TyCtxt<'tcx>,
    always_live_locals: DenseBitSet<Local>,
}
