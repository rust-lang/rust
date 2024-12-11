use std::iter;

use itertools::Itertools as _;
use rustc_abi::Align;
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, ConstCodegenMethods, StaticCodegenMethods,
};
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::IndexVec;
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_session::RemapFileNameExt;
use rustc_session::config::RemapPathScopeComponents;
use rustc_span::def_id::DefIdSet;
use rustc_span::{Span, Symbol};
use tracing::debug;

use crate::common::CodegenCx;
use crate::coverageinfo::llvm_cov;
use crate::coverageinfo::map_data::FunctionCoverage;
use crate::coverageinfo::mapgen::covfun::prepare_covfun_record;
use crate::llvm;

mod covfun;

/// Generates and exports the coverage map, which is embedded in special
/// linker sections in the final binary.
///
/// Those sections are then read and understood by LLVM's `llvm-cov` tool,
/// which is distributed in the `llvm-tools` rustup component.
pub(crate) fn finalize(cx: &CodegenCx<'_, '_>) {
    let tcx = cx.tcx;

    // Ensure that LLVM is using a version of the coverage mapping format that
    // agrees with our Rust-side code. Expected versions (encoded as n-1) are:
    // - `CovMapVersion::Version7` (6) used by LLVM 18-19
    let covmap_version = {
        let llvm_covmap_version = llvm_cov::mapping_version();
        let expected_versions = 6..=6;
        assert!(
            expected_versions.contains(&llvm_covmap_version),
            "Coverage mapping version exposed by `llvm-wrapper` is out of sync; \
            expected {expected_versions:?} but was {llvm_covmap_version}"
        );
        // This is the version number that we will embed in the covmap section:
        llvm_covmap_version
    };

    debug!("Generating coverage map for CodegenUnit: `{}`", cx.codegen_unit.name());

    // In order to show that unused functions have coverage counts of zero (0), LLVM requires the
    // functions exist. Generate synthetic functions with a (required) single counter, and add the
    // MIR `Coverage` code regions to the `function_coverage_map`, before calling
    // `ctx.take_function_coverage_map()`.
    if cx.codegen_unit.is_code_coverage_dead_code_cgu() {
        add_unused_functions(cx);
    }

    // FIXME(#132395): Can this be none even when coverage is enabled?
    let function_coverage_map = match cx.coverage_cx {
        Some(ref cx) => cx.take_function_coverage_map(),
        None => return,
    };
    if function_coverage_map.is_empty() {
        // This CGU has no functions with coverage instrumentation.
        return;
    }

    let all_file_names = function_coverage_map
        .iter()
        .map(|(_, fn_cov)| fn_cov.function_coverage_info.body_span)
        .map(|span| span_file_name(tcx, span));
    let global_file_table = GlobalFileTable::new(all_file_names);

    // Encode all filenames referenced by coverage mappings in this CGU.
    let filenames_buffer = global_file_table.make_filenames_buffer(tcx);

    let filenames_size = filenames_buffer.len();
    let filenames_val = cx.const_bytes(&filenames_buffer);
    let filenames_ref = llvm_cov::hash_bytes(&filenames_buffer);

    let mut unused_function_names = Vec::new();

    let covfun_records = function_coverage_map
        .into_iter()
        .filter_map(|(instance, function_coverage)| {
            prepare_covfun_record(tcx, &global_file_table, instance, &function_coverage)
        })
        .collect::<Vec<_>>();

    // If there are no covfun records for this CGU, don't generate a covmap record.
    // Emitting a covmap record without any covfun records causes `llvm-cov` to
    // fail when generating coverage reports, and if there are no covfun records
    // then the covmap record isn't useful anyway.
    // This should prevent a repeat of <https://github.com/rust-lang/rust/issues/133606>.
    if covfun_records.is_empty() {
        return;
    }

    for covfun in &covfun_records {
        unused_function_names.extend(covfun.mangled_function_name_if_unused());

        covfun::generate_covfun_record(cx, filenames_ref, covfun)
    }

    // For unused functions, we need to take their mangled names and store them
    // in a specially-named global array. LLVM's `InstrProfiling` pass will
    // detect this global and include those names in its `__llvm_prf_names`
    // section. (See `llvm/lib/Transforms/Instrumentation/InstrProfiling.cpp`.)
    if !unused_function_names.is_empty() {
        assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());

        let name_globals = unused_function_names
            .into_iter()
            .map(|mangled_function_name| cx.const_str(mangled_function_name).0)
            .collect::<Vec<_>>();
        let initializer = cx.const_array(cx.type_ptr(), &name_globals);

        let array = llvm::add_global(cx.llmod, cx.val_ty(initializer), c"__llvm_coverage_names");
        llvm::set_global_constant(array, true);
        llvm::set_linkage(array, llvm::Linkage::InternalLinkage);
        llvm::set_initializer(array, initializer);
    }

    // Generate the coverage map header, which contains the filenames used by
    // this CGU's coverage mappings, and store it in a well-known global.
    // (This is skipped if we returned early due to having no covfun records.)
    generate_covmap_record(cx, covmap_version, filenames_size, filenames_val);
}

/// Maps "global" (per-CGU) file ID numbers to their underlying filenames.
struct GlobalFileTable {
    /// This "raw" table doesn't include the working dir, so a filename's
    /// global ID is its index in this set **plus one**.
    raw_file_table: FxIndexSet<Symbol>,
}

impl GlobalFileTable {
    fn new(all_file_names: impl IntoIterator<Item = Symbol>) -> Self {
        // Collect all of the filenames into a set. Filenames usually come in
        // contiguous runs, so we can dedup adjacent ones to save work.
        let mut raw_file_table = all_file_names.into_iter().dedup().collect::<FxIndexSet<Symbol>>();

        // Sort the file table by its actual string values, not the arbitrary
        // ordering of its symbols.
        raw_file_table.sort_unstable_by(|a, b| a.as_str().cmp(b.as_str()));

        Self { raw_file_table }
    }

    fn global_file_id_for_file_name(&self, file_name: Symbol) -> GlobalFileId {
        let raw_id = self.raw_file_table.get_index_of(&file_name).unwrap_or_else(|| {
            bug!("file name not found in prepared global file table: {file_name}");
        });
        // The raw file table doesn't include an entry for the working dir
        // (which has ID 0), so add 1 to get the correct ID.
        GlobalFileId::from_usize(raw_id + 1)
    }

    fn make_filenames_buffer(&self, tcx: TyCtxt<'_>) -> Vec<u8> {
        // LLVM Coverage Mapping Format version 6 (zero-based encoded as 5)
        // requires setting the first filename to the compilation directory.
        // Since rustc generates coverage maps with relative paths, the
        // compilation directory can be combined with the relative paths
        // to get absolute paths, if needed.
        use rustc_session::RemapFileNameExt;
        use rustc_session::config::RemapPathScopeComponents;
        let working_dir: &str = &tcx
            .sess
            .opts
            .working_dir
            .for_scope(tcx.sess, RemapPathScopeComponents::MACRO)
            .to_string_lossy();

        // Insert the working dir at index 0, before the other filenames.
        let filenames =
            iter::once(working_dir).chain(self.raw_file_table.iter().map(Symbol::as_str));
        llvm_cov::write_filenames_to_buffer(filenames)
    }
}

rustc_index::newtype_index! {
    /// An index into the CGU's overall list of file paths. The underlying paths
    /// will be embedded in the `__llvm_covmap` linker section.
    struct GlobalFileId {}
}
rustc_index::newtype_index! {
    /// An index into a function's list of global file IDs. That underlying list
    /// of local-to-global mappings will be embedded in the function's record in
    /// the `__llvm_covfun` linker section.
    pub(crate) struct LocalFileId {}
}

/// Holds a mapping from "local" (per-function) file IDs to "global" (per-CGU)
/// file IDs.
#[derive(Debug, Default)]
struct VirtualFileMapping {
    local_to_global: IndexVec<LocalFileId, GlobalFileId>,
    global_to_local: FxIndexMap<GlobalFileId, LocalFileId>,
}

impl VirtualFileMapping {
    fn local_id_for_global(&mut self, global_file_id: GlobalFileId) -> LocalFileId {
        *self
            .global_to_local
            .entry(global_file_id)
            .or_insert_with(|| self.local_to_global.push(global_file_id))
    }

    fn to_vec(&self) -> Vec<u32> {
        // This clone could be avoided by transmuting `&[GlobalFileId]` to `&[u32]`,
        // but it isn't hot or expensive enough to justify the extra unsafety.
        self.local_to_global.iter().map(|&global| GlobalFileId::as_u32(global)).collect()
    }
}

fn span_file_name(tcx: TyCtxt<'_>, span: Span) -> Symbol {
    let source_file = tcx.sess.source_map().lookup_source_file(span.lo());
    let name =
        source_file.name.for_scope(tcx.sess, RemapPathScopeComponents::MACRO).to_string_lossy();
    Symbol::intern(&name)
}

/// Generates the contents of the covmap record for this CGU, which mostly
/// consists of a header and a list of filenames. The record is then stored
/// as a global variable in the `__llvm_covmap` section.
fn generate_covmap_record<'ll>(
    cx: &CodegenCx<'ll, '_>,
    version: u32,
    filenames_size: usize,
    filenames_val: &'ll llvm::Value,
) {
    debug!("cov map: filenames_size = {}, 0-based version = {}", filenames_size, version);

    // Create the coverage data header (Note, fields 0 and 2 are now always zero,
    // as of `llvm::coverage::CovMapVersion::Version4`.)
    let zero_was_n_records_val = cx.const_u32(0);
    let filenames_size_val = cx.const_u32(filenames_size as u32);
    let zero_was_coverage_size_val = cx.const_u32(0);
    let version_val = cx.const_u32(version);
    let cov_data_header_val = cx.const_struct(
        &[zero_was_n_records_val, filenames_size_val, zero_was_coverage_size_val, version_val],
        /*packed=*/ false,
    );

    // Create the complete LLVM coverage data value to add to the LLVM IR
    let covmap_data =
        cx.const_struct(&[cov_data_header_val, filenames_val], /*packed=*/ false);

    let llglobal = llvm::add_global(cx.llmod, cx.val_ty(covmap_data), &llvm_cov::covmap_var_name());
    llvm::set_initializer(llglobal, covmap_data);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, llvm::Linkage::PrivateLinkage);
    llvm::set_section(llglobal, &llvm_cov::covmap_section_name(cx.llmod));
    // LLVM's coverage mapping format specifies 8-byte alignment for items in this section.
    // <https://llvm.org/docs/CoverageMappingFormat.html>
    llvm::set_alignment(llglobal, Align::EIGHT);
    cx.add_used_global(llglobal);
}

/// Each CGU will normally only emit coverage metadata for the functions that it actually generates.
/// But since we don't want unused functions to disappear from coverage reports, we also scan for
/// functions that were instrumented but are not participating in codegen.
///
/// These unused functions don't need to be codegenned, but we do need to add them to the function
/// coverage map (in a single designated CGU) so that we still emit coverage mappings for them.
/// We also end up adding their symbol names to a special global array that LLVM will include in
/// its embedded coverage data.
fn add_unused_functions(cx: &CodegenCx<'_, '_>) {
    assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());

    let tcx = cx.tcx;
    let usage = prepare_usage_sets(tcx);

    let is_unused_fn = |def_id: LocalDefId| -> bool {
        let def_id = def_id.to_def_id();

        // To be eligible for "unused function" mappings, a definition must:
        // - Be function-like
        // - Not participate directly in codegen (or have lost all its coverage statements)
        // - Not have any coverage statements inlined into codegenned functions
        tcx.def_kind(def_id).is_fn_like()
            && (!usage.all_mono_items.contains(&def_id)
                || usage.missing_own_coverage.contains(&def_id))
            && !usage.used_via_inlining.contains(&def_id)
    };

    // Scan for unused functions that were instrumented for coverage.
    for def_id in tcx.mir_keys(()).iter().copied().filter(|&def_id| is_unused_fn(def_id)) {
        // Get the coverage info from MIR, skipping functions that were never instrumented.
        let body = tcx.optimized_mir(def_id);
        let Some(function_coverage_info) = body.function_coverage_info.as_deref() else { continue };

        // FIXME(79651): Consider trying to filter out dummy instantiations of
        // unused generic functions from library crates, because they can produce
        // "unused instantiation" in coverage reports even when they are actually
        // used by some downstream crate in the same binary.

        debug!("generating unused fn: {def_id:?}");
        add_unused_function_coverage(cx, def_id, function_coverage_info);
    }
}

struct UsageSets<'tcx> {
    all_mono_items: &'tcx DefIdSet,
    used_via_inlining: FxHashSet<DefId>,
    missing_own_coverage: FxHashSet<DefId>,
}

/// Prepare sets of definitions that are relevant to deciding whether something
/// is an "unused function" for coverage purposes.
fn prepare_usage_sets<'tcx>(tcx: TyCtxt<'tcx>) -> UsageSets<'tcx> {
    let (all_mono_items, cgus) = tcx.collect_and_partition_mono_items(());

    // Obtain a MIR body for each function participating in codegen, via an
    // arbitrary instance.
    let mut def_ids_seen = FxHashSet::default();
    let def_and_mir_for_all_mono_fns = cgus
        .iter()
        .flat_map(|cgu| cgu.items().keys())
        .filter_map(|item| match item {
            mir::mono::MonoItem::Fn(instance) => Some(instance),
            mir::mono::MonoItem::Static(_) | mir::mono::MonoItem::GlobalAsm(_) => None,
        })
        // We only need one arbitrary instance per definition.
        .filter(move |instance| def_ids_seen.insert(instance.def_id()))
        .map(|instance| {
            // We don't care about the instance, just its underlying MIR.
            let body = tcx.instance_mir(instance.def);
            (instance.def_id(), body)
        });

    // Functions whose coverage statements were found inlined into other functions.
    let mut used_via_inlining = FxHashSet::default();
    // Functions that were instrumented, but had all of their coverage statements
    // removed by later MIR transforms (e.g. UnreachablePropagation).
    let mut missing_own_coverage = FxHashSet::default();

    for (def_id, body) in def_and_mir_for_all_mono_fns {
        let mut saw_own_coverage = false;

        // Inspect every coverage statement in the function's MIR.
        for stmt in body
            .basic_blocks
            .iter()
            .flat_map(|block| &block.statements)
            .filter(|stmt| matches!(stmt.kind, mir::StatementKind::Coverage(_)))
        {
            if let Some(inlined) = stmt.source_info.scope.inlined_instance(&body.source_scopes) {
                // This coverage statement was inlined from another function.
                used_via_inlining.insert(inlined.def_id());
            } else {
                // Non-inlined coverage statements belong to the enclosing function.
                saw_own_coverage = true;
            }
        }

        if !saw_own_coverage && body.function_coverage_info.is_some() {
            missing_own_coverage.insert(def_id);
        }
    }

    UsageSets { all_mono_items, used_via_inlining, missing_own_coverage }
}

fn add_unused_function_coverage<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    def_id: LocalDefId,
    function_coverage_info: &'tcx mir::coverage::FunctionCoverageInfo,
) {
    let tcx = cx.tcx;
    let def_id = def_id.to_def_id();

    // Make a dummy instance that fills in all generics with placeholders.
    let instance = ty::Instance::new(
        def_id,
        ty::GenericArgs::for_item(tcx, def_id, |param, _| {
            if let ty::GenericParamDefKind::Lifetime = param.kind {
                tcx.lifetimes.re_erased.into()
            } else {
                tcx.mk_param_from_def(param)
            }
        }),
    );

    // An unused function's mappings will all be rewritten to map to zero.
    let function_coverage = FunctionCoverage::new_unused(function_coverage_info);
    cx.coverage_cx().function_coverage_map.borrow_mut().insert(instance, function_coverage);
}
