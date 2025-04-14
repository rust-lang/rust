use std::collections::BTreeSet;
use std::fmt::{Display, Write as _};
use std::path::{Path, PathBuf};
use std::{fs, io};

use rustc_abi::Size;
use rustc_ast::InlineAsmTemplatePiece;
use tracing::trace;
use ty::print::PrettyPrinter;

use super::graphviz::write_mir_fn_graphviz;
use crate::mir::interpret::{
    AllocBytes, AllocId, Allocation, ConstAllocation, GlobalAlloc, Pointer, Provenance,
    alloc_range, read_target_uint,
};
use crate::mir::visit::Visitor;
use crate::mir::*;

const INDENT: &str = "    ";
/// Alignment for lining up comments following MIR statements
pub(crate) const ALIGN: usize = 40;

/// An indication of where we are in the control flow graph. Used for printing
/// extra information in `dump_mir`
#[derive(Clone, Copy)]
pub enum PassWhere {
    /// We have not started dumping the control flow graph, but we are about to.
    BeforeCFG,

    /// We just finished dumping the control flow graph. This is right before EOF
    AfterCFG,

    /// We are about to start dumping the given basic block.
    BeforeBlock(BasicBlock),

    /// We are just about to dump the given statement or terminator.
    BeforeLocation(Location),

    /// We just dumped the given statement or terminator.
    AfterLocation(Location),

    /// We just dumped the terminator for a block but not the closing `}`.
    AfterTerminator(BasicBlock),
}

/// Cosmetic options for pretty-printing the MIR contents, gathered from the CLI. Each pass can
/// override these when dumping its own specific MIR information with [`dump_mir_with_options`].
#[derive(Copy, Clone)]
pub struct PrettyPrintMirOptions {
    /// Whether to include extra comments, like span info. From `-Z mir-include-spans`.
    pub include_extra_comments: bool,
}

impl PrettyPrintMirOptions {
    /// Create the default set of MIR pretty-printing options from the CLI flags.
    pub fn from_cli(tcx: TyCtxt<'_>) -> Self {
        Self { include_extra_comments: tcx.sess.opts.unstable_opts.mir_include_spans.is_enabled() }
    }
}

/// If the session is properly configured, dumps a human-readable representation of the MIR (with
/// default pretty-printing options) into:
///
/// ```text
/// rustc.node<node_id>.<pass_num>.<pass_name>.<disambiguator>
/// ```
///
/// Output from this function is controlled by passing `-Z dump-mir=<filter>`,
/// where `<filter>` takes the following forms:
///
/// - `all` -- dump MIR for all fns, all passes, all everything
/// - a filter defined by a set of substrings combined with `&` and `|`
///   (`&` has higher precedence). At least one of the `|`-separated groups
///   must match; an `|`-separated group matches if all of its `&`-separated
///   substrings are matched.
///
/// Example:
///
/// - `nll` == match if `nll` appears in the name
/// - `foo & nll` == match if `foo` and `nll` both appear in the name
/// - `foo & nll | typeck` == match if `foo` and `nll` both appear in the name
///   or `typeck` appears in the name.
/// - `foo & nll | bar & typeck` == match if `foo` and `nll` both appear in the name
///   or `typeck` and `bar` both appear in the name.
#[inline]
pub fn dump_mir<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    pass_num: bool,
    pass_name: &str,
    disambiguator: &dyn Display,
    body: &Body<'tcx>,
    extra_data: F,
) where
    F: FnMut(PassWhere, &mut dyn io::Write) -> io::Result<()>,
{
    dump_mir_with_options(
        tcx,
        pass_num,
        pass_name,
        disambiguator,
        body,
        extra_data,
        PrettyPrintMirOptions::from_cli(tcx),
    );
}

/// If the session is properly configured, dumps a human-readable representation of the MIR, with
/// the given [pretty-printing options][PrettyPrintMirOptions].
///
/// See [`dump_mir`] for more details.
///
#[inline]
pub fn dump_mir_with_options<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    pass_num: bool,
    pass_name: &str,
    disambiguator: &dyn Display,
    body: &Body<'tcx>,
    extra_data: F,
    options: PrettyPrintMirOptions,
) where
    F: FnMut(PassWhere, &mut dyn io::Write) -> io::Result<()>,
{
    if !dump_enabled(tcx, pass_name, body.source.def_id()) {
        return;
    }

    dump_matched_mir_node(tcx, pass_num, pass_name, disambiguator, body, extra_data, options);
}

pub fn dump_enabled(tcx: TyCtxt<'_>, pass_name: &str, def_id: DefId) -> bool {
    let Some(ref filters) = tcx.sess.opts.unstable_opts.dump_mir else {
        return false;
    };
    // see notes on #41697 below
    let node_path = ty::print::with_forced_impl_filename_line!(tcx.def_path_str(def_id));
    filters.split('|').any(|or_filter| {
        or_filter.split('&').all(|and_filter| {
            let and_filter_trimmed = and_filter.trim();
            and_filter_trimmed == "all"
                || pass_name.contains(and_filter_trimmed)
                || node_path.contains(and_filter_trimmed)
        })
    })
}

// #41697 -- we use `with_forced_impl_filename_line()` because
// `def_path_str()` would otherwise trigger `type_of`, and this can
// run while we are already attempting to evaluate `type_of`.

/// Most use-cases of dumping MIR should use the [dump_mir] entrypoint instead, which will also
/// check if dumping MIR is enabled, and if this body matches the filters passed on the CLI.
///
/// That being said, if the above requirements have been validated already, this function is where
/// most of the MIR dumping occurs, if one needs to export it to a file they have created with
/// [create_dump_file], rather than to a new file created as part of [dump_mir], or to stdout/stderr
/// for debugging purposes.
pub fn dump_mir_to_writer<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    pass_name: &str,
    disambiguator: &dyn Display,
    body: &Body<'tcx>,
    w: &mut dyn io::Write,
    mut extra_data: F,
    options: PrettyPrintMirOptions,
) -> io::Result<()>
where
    F: FnMut(PassWhere, &mut dyn io::Write) -> io::Result<()>,
{
    // see notes on #41697 above
    let def_path =
        ty::print::with_forced_impl_filename_line!(tcx.def_path_str(body.source.def_id()));
    // ignore-tidy-odd-backticks the literal below is fine
    write!(w, "// MIR for `{def_path}")?;
    match body.source.promoted {
        None => write!(w, "`")?,
        Some(promoted) => write!(w, "::{promoted:?}`")?,
    }
    writeln!(w, " {disambiguator} {pass_name}")?;
    if let Some(ref layout) = body.coroutine_layout_raw() {
        writeln!(w, "/* coroutine_layout = {layout:#?} */")?;
    }
    writeln!(w)?;
    extra_data(PassWhere::BeforeCFG, w)?;
    write_user_type_annotations(tcx, body, w)?;
    write_mir_fn(tcx, body, &mut extra_data, w, options)?;
    extra_data(PassWhere::AfterCFG, w)
}

fn dump_matched_mir_node<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    pass_num: bool,
    pass_name: &str,
    disambiguator: &dyn Display,
    body: &Body<'tcx>,
    extra_data: F,
    options: PrettyPrintMirOptions,
) where
    F: FnMut(PassWhere, &mut dyn io::Write) -> io::Result<()>,
{
    let _: io::Result<()> = try {
        let mut file = create_dump_file(tcx, "mir", pass_num, pass_name, disambiguator, body)?;
        dump_mir_to_writer(tcx, pass_name, disambiguator, body, &mut file, extra_data, options)?;
    };

    if tcx.sess.opts.unstable_opts.dump_mir_graphviz {
        let _: io::Result<()> = try {
            let mut file = create_dump_file(tcx, "dot", pass_num, pass_name, disambiguator, body)?;
            write_mir_fn_graphviz(tcx, body, false, &mut file)?;
        };
    }
}

/// Returns the path to the filename where we should dump a given MIR.
/// Also used by other bits of code (e.g., NLL inference) that dump
/// graphviz data or other things.
fn dump_path<'tcx>(
    tcx: TyCtxt<'tcx>,
    extension: &str,
    pass_num: bool,
    pass_name: &str,
    disambiguator: &dyn Display,
    body: &Body<'tcx>,
) -> PathBuf {
    let source = body.source;
    let promotion_id = match source.promoted {
        Some(id) => format!("-{id:?}"),
        None => String::new(),
    };

    let pass_num = if tcx.sess.opts.unstable_opts.dump_mir_exclude_pass_number {
        String::new()
    } else if pass_num {
        let (dialect_index, phase_index) = body.phase.index();
        format!(".{}-{}-{:03}", dialect_index, phase_index, body.pass_count)
    } else {
        ".-------".to_string()
    };

    let crate_name = tcx.crate_name(source.def_id().krate);
    let item_name = tcx.def_path(source.def_id()).to_filename_friendly_no_crate();
    // All drop shims have the same DefId, so we have to add the type
    // to get unique file names.
    let shim_disambiguator = match source.instance {
        ty::InstanceKind::DropGlue(_, Some(ty)) => {
            // Unfortunately, pretty-printed typed are not very filename-friendly.
            // We dome some filtering.
            let mut s = ".".to_owned();
            s.extend(ty.to_string().chars().filter_map(|c| match c {
                ' ' => None,
                ':' | '<' | '>' => Some('_'),
                c => Some(c),
            }));
            s
        }
        ty::InstanceKind::AsyncDropGlueCtorShim(_, Some(ty)) => {
            // Unfortunately, pretty-printed typed are not very filename-friendly.
            // We dome some filtering.
            let mut s = ".".to_owned();
            s.extend(ty.to_string().chars().filter_map(|c| match c {
                ' ' => None,
                ':' | '<' | '>' => Some('_'),
                c => Some(c),
            }));
            s
        }
        _ => String::new(),
    };

    let mut file_path = PathBuf::new();
    file_path.push(Path::new(&tcx.sess.opts.unstable_opts.dump_mir_dir));

    let file_name = format!(
        "{crate_name}.{item_name}{shim_disambiguator}{promotion_id}{pass_num}.{pass_name}.{disambiguator}.{extension}",
    );

    file_path.push(&file_name);

    file_path
}

/// Attempts to open a file where we should dump a given MIR or other
/// bit of MIR-related data. Used by `mir-dump`, but also by other
/// bits of code (e.g., NLL inference) that dump graphviz data or
/// other things, and hence takes the extension as an argument.
pub fn create_dump_file<'tcx>(
    tcx: TyCtxt<'tcx>,
    extension: &str,
    pass_num: bool,
    pass_name: &str,
    disambiguator: &dyn Display,
    body: &Body<'tcx>,
) -> io::Result<io::BufWriter<fs::File>> {
    let file_path = dump_path(tcx, extension, pass_num, pass_name, disambiguator, body);
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("IO error creating MIR dump directory: {parent:?}; {e}"),
            )
        })?;
    }
    fs::File::create_buffered(&file_path).map_err(|e| {
        io::Error::new(e.kind(), format!("IO error creating MIR dump file: {file_path:?}; {e}"))
    })
}

///////////////////////////////////////////////////////////////////////////
// Whole MIR bodies

/// Write out a human-readable textual representation for the given MIR, with the default
/// [PrettyPrintMirOptions].
pub fn write_mir_pretty<'tcx>(
    tcx: TyCtxt<'tcx>,
    single: Option<DefId>,
    w: &mut dyn io::Write,
) -> io::Result<()> {
    let options = PrettyPrintMirOptions::from_cli(tcx);

    writeln!(w, "// WARNING: This output format is intended for human consumers only")?;
    writeln!(w, "// and is subject to change without notice. Knock yourself out.")?;
    writeln!(w, "// HINT: See also -Z dump-mir for MIR at specific points during compilation.")?;

    let mut first = true;
    for def_id in dump_mir_def_ids(tcx, single) {
        if first {
            first = false;
        } else {
            // Put empty lines between all items
            writeln!(w)?;
        }

        let render_body = |w: &mut dyn io::Write, body| -> io::Result<()> {
            write_mir_fn(tcx, body, &mut |_, _| Ok(()), w, options)?;

            for body in tcx.promoted_mir(def_id) {
                writeln!(w)?;
                write_mir_fn(tcx, body, &mut |_, _| Ok(()), w, options)?;
            }
            Ok(())
        };

        // For `const fn` we want to render both the optimized MIR and the MIR for ctfe.
        if tcx.is_const_fn(def_id) {
            render_body(w, tcx.optimized_mir(def_id))?;
            writeln!(w)?;
            writeln!(w, "// MIR FOR CTFE")?;
            // Do not use `render_body`, as that would render the promoteds again, but these
            // are shared between mir_for_ctfe and optimized_mir
            write_mir_fn(tcx, tcx.mir_for_ctfe(def_id), &mut |_, _| Ok(()), w, options)?;
        } else {
            let instance_mir = tcx.instance_mir(ty::InstanceKind::Item(def_id));
            render_body(w, instance_mir)?;
        }
    }
    Ok(())
}

/// Write out a human-readable textual representation for the given function.
pub fn write_mir_fn<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    extra_data: &mut F,
    w: &mut dyn io::Write,
    options: PrettyPrintMirOptions,
) -> io::Result<()>
where
    F: FnMut(PassWhere, &mut dyn io::Write) -> io::Result<()>,
{
    write_mir_intro(tcx, body, w, options)?;
    for block in body.basic_blocks.indices() {
        extra_data(PassWhere::BeforeBlock(block), w)?;
        write_basic_block(tcx, block, body, extra_data, w, options)?;
        if block.index() + 1 != body.basic_blocks.len() {
            writeln!(w)?;
        }
    }

    writeln!(w, "}}")?;

    write_allocations(tcx, body, w)?;

    Ok(())
}

/// Prints local variables in a scope tree.
fn write_scope_tree(
    tcx: TyCtxt<'_>,
    body: &Body<'_>,
    scope_tree: &FxHashMap<SourceScope, Vec<SourceScope>>,
    w: &mut dyn io::Write,
    parent: SourceScope,
    depth: usize,
    options: PrettyPrintMirOptions,
) -> io::Result<()> {
    let indent = depth * INDENT.len();

    // Local variable debuginfo.
    for var_debug_info in &body.var_debug_info {
        if var_debug_info.source_info.scope != parent {
            // Not declared in this scope.
            continue;
        }

        let indented_debug_info = format!("{0:1$}debug {2:?};", INDENT, indent, var_debug_info);

        if options.include_extra_comments {
            writeln!(
                w,
                "{0:1$} // in {2}",
                indented_debug_info,
                ALIGN,
                comment(tcx, var_debug_info.source_info),
            )?;
        } else {
            writeln!(w, "{indented_debug_info}")?;
        }
    }

    // Local variable types.
    for (local, local_decl) in body.local_decls.iter_enumerated() {
        if (1..body.arg_count + 1).contains(&local.index()) {
            // Skip over argument locals, they're printed in the signature.
            continue;
        }

        if local_decl.source_info.scope != parent {
            // Not declared in this scope.
            continue;
        }

        let mut_str = local_decl.mutability.prefix_str();

        let mut indented_decl = ty::print::with_no_trimmed_paths!(format!(
            "{0:1$}let {2}{3:?}: {4}",
            INDENT, indent, mut_str, local, local_decl.ty
        ));
        if let Some(user_ty) = &local_decl.user_ty {
            for user_ty in user_ty.projections() {
                write!(indented_decl, " as {user_ty:?}").unwrap();
            }
        }
        indented_decl.push(';');

        let local_name = if local == RETURN_PLACE { " return place" } else { "" };

        if options.include_extra_comments {
            writeln!(
                w,
                "{0:1$} //{2} in {3}",
                indented_decl,
                ALIGN,
                local_name,
                comment(tcx, local_decl.source_info),
            )?;
        } else {
            writeln!(w, "{indented_decl}",)?;
        }
    }

    let Some(children) = scope_tree.get(&parent) else {
        return Ok(());
    };

    for &child in children {
        let child_data = &body.source_scopes[child];
        assert_eq!(child_data.parent_scope, Some(parent));

        let (special, span) = if let Some((callee, callsite_span)) = child_data.inlined {
            (
                format!(
                    " (inlined {}{})",
                    if callee.def.requires_caller_location(tcx) { "#[track_caller] " } else { "" },
                    callee
                ),
                Some(callsite_span),
            )
        } else {
            (String::new(), None)
        };

        let indented_header = format!("{0:1$}scope {2}{3} {{", "", indent, child.index(), special);

        if options.include_extra_comments {
            if let Some(span) = span {
                writeln!(
                    w,
                    "{0:1$} // at {2}",
                    indented_header,
                    ALIGN,
                    tcx.sess.source_map().span_to_embeddable_string(span),
                )?;
            } else {
                writeln!(w, "{indented_header}")?;
            }
        } else {
            writeln!(w, "{indented_header}")?;
        }

        write_scope_tree(tcx, body, scope_tree, w, child, depth + 1, options)?;
        writeln!(w, "{0:1$}}}", "", depth * INDENT.len())?;
    }

    Ok(())
}

impl Debug for VarDebugInfo<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        if let Some(box VarDebugInfoFragment { ty, ref projection }) = self.composite {
            pre_fmt_projection(&projection[..], fmt)?;
            write!(fmt, "({}: {})", self.name, ty)?;
            post_fmt_projection(&projection[..], fmt)?;
        } else {
            write!(fmt, "{}", self.name)?;
        }

        write!(fmt, " => {:?}", self.value)
    }
}

/// Write out a human-readable textual representation of the MIR's `fn` type and the types of its
/// local variables (both user-defined bindings and compiler temporaries).
fn write_mir_intro<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'_>,
    w: &mut dyn io::Write,
    options: PrettyPrintMirOptions,
) -> io::Result<()> {
    write_mir_sig(tcx, body, w)?;
    writeln!(w, "{{")?;

    // construct a scope tree and write it out
    let mut scope_tree: FxHashMap<SourceScope, Vec<SourceScope>> = Default::default();
    for (index, scope_data) in body.source_scopes.iter_enumerated() {
        if let Some(parent) = scope_data.parent_scope {
            scope_tree.entry(parent).or_default().push(index);
        } else {
            // Only the argument scope has no parent, because it's the root.
            assert_eq!(index, OUTERMOST_SOURCE_SCOPE);
        }
    }

    write_scope_tree(tcx, body, &scope_tree, w, OUTERMOST_SOURCE_SCOPE, 1, options)?;

    // Add an empty line before the first block is printed.
    writeln!(w)?;

    if let Some(coverage_info_hi) = &body.coverage_info_hi {
        write_coverage_info_hi(coverage_info_hi, w)?;
    }
    if let Some(function_coverage_info) = &body.function_coverage_info {
        write_function_coverage_info(function_coverage_info, w)?;
    }

    Ok(())
}

fn write_coverage_info_hi(
    coverage_info_hi: &coverage::CoverageInfoHi,
    w: &mut dyn io::Write,
) -> io::Result<()> {
    let coverage::CoverageInfoHi {
        num_block_markers: _,
        branch_spans,
        mcdc_degraded_branch_spans,
        mcdc_spans,
    } = coverage_info_hi;

    // Only add an extra trailing newline if we printed at least one thing.
    let mut did_print = false;

    for coverage::BranchSpan { span, true_marker, false_marker } in branch_spans {
        writeln!(
            w,
            "{INDENT}coverage branch {{ true: {true_marker:?}, false: {false_marker:?} }} => {span:?}",
        )?;
        did_print = true;
    }

    for coverage::MCDCBranchSpan { span, true_marker, false_marker, .. } in
        mcdc_degraded_branch_spans
    {
        writeln!(
            w,
            "{INDENT}coverage branch {{ true: {true_marker:?}, false: {false_marker:?} }} => {span:?}",
        )?;
        did_print = true;
    }

    for (
        coverage::MCDCDecisionSpan { span, end_markers, decision_depth, num_conditions: _ },
        conditions,
    ) in mcdc_spans
    {
        let num_conditions = conditions.len();
        writeln!(
            w,
            "{INDENT}coverage mcdc decision {{ num_conditions: {num_conditions:?}, end: {end_markers:?}, depth: {decision_depth:?} }} => {span:?}"
        )?;
        for coverage::MCDCBranchSpan { span, condition_info, true_marker, false_marker } in
            conditions
        {
            writeln!(
                w,
                "{INDENT}coverage mcdc branch {{ condition_id: {:?}, true: {true_marker:?}, false: {false_marker:?} }} => {span:?}",
                condition_info.condition_id
            )?;
        }
        did_print = true;
    }

    if did_print {
        writeln!(w)?;
    }

    Ok(())
}

fn write_function_coverage_info(
    function_coverage_info: &coverage::FunctionCoverageInfo,
    w: &mut dyn io::Write,
) -> io::Result<()> {
    let coverage::FunctionCoverageInfo { mappings, .. } = function_coverage_info;

    for coverage::Mapping { kind, span } in mappings {
        writeln!(w, "{INDENT}coverage {kind:?} => {span:?};")?;
    }
    writeln!(w)?;

    Ok(())
}

fn write_mir_sig(tcx: TyCtxt<'_>, body: &Body<'_>, w: &mut dyn io::Write) -> io::Result<()> {
    use rustc_hir::def::DefKind;

    trace!("write_mir_sig: {:?}", body.source.instance);
    let def_id = body.source.def_id();
    let kind = tcx.def_kind(def_id);
    let is_function = match kind {
        DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(..) | DefKind::SyntheticCoroutineBody => {
            true
        }
        _ => tcx.is_closure_like(def_id),
    };
    match (kind, body.source.promoted) {
        (_, Some(_)) => write!(w, "const ")?, // promoteds are the closest to consts
        (DefKind::Const | DefKind::AssocConst, _) => write!(w, "const ")?,
        (DefKind::Static { safety: _, mutability: hir::Mutability::Not, nested: false }, _) => {
            write!(w, "static ")?
        }
        (DefKind::Static { safety: _, mutability: hir::Mutability::Mut, nested: false }, _) => {
            write!(w, "static mut ")?
        }
        (_, _) if is_function => write!(w, "fn ")?,
        // things like anon const, not an item
        (DefKind::AnonConst | DefKind::InlineConst, _) => {}
        // `global_asm!` have fake bodies, which we may dump after mir-build
        (DefKind::GlobalAsm, _) => {}
        _ => bug!("Unexpected def kind {:?}", kind),
    }

    ty::print::with_forced_impl_filename_line! {
        // see notes on #41697 elsewhere
        write!(w, "{}", tcx.def_path_str(def_id))?
    }
    if let Some(p) = body.source.promoted {
        write!(w, "::{p:?}")?;
    }

    if body.source.promoted.is_none() && is_function {
        write!(w, "(")?;

        // fn argument types.
        for (i, arg) in body.args_iter().enumerate() {
            if i != 0 {
                write!(w, ", ")?;
            }
            write!(w, "{:?}: {}", Place::from(arg), body.local_decls[arg].ty)?;
        }

        write!(w, ") -> {}", body.return_ty())?;
    } else {
        assert_eq!(body.arg_count, 0);
        write!(w, ": {} =", body.return_ty())?;
    }

    if let Some(yield_ty) = body.yield_ty() {
        writeln!(w)?;
        writeln!(w, "yields {yield_ty}")?;
    }

    write!(w, " ")?;
    // Next thing that gets printed is the opening {

    Ok(())
}

fn write_user_type_annotations(
    tcx: TyCtxt<'_>,
    body: &Body<'_>,
    w: &mut dyn io::Write,
) -> io::Result<()> {
    if !body.user_type_annotations.is_empty() {
        writeln!(w, "| User Type Annotations")?;
    }
    for (index, annotation) in body.user_type_annotations.iter_enumerated() {
        writeln!(
            w,
            "| {:?}: user_ty: {}, span: {}, inferred_ty: {}",
            index.index(),
            annotation.user_ty,
            tcx.sess.source_map().span_to_embeddable_string(annotation.span),
            with_no_trimmed_paths!(format!("{}", annotation.inferred_ty)),
        )?;
    }
    if !body.user_type_annotations.is_empty() {
        writeln!(w, "|")?;
    }
    Ok(())
}

pub fn dump_mir_def_ids(tcx: TyCtxt<'_>, single: Option<DefId>) -> Vec<DefId> {
    if let Some(i) = single {
        vec![i]
    } else {
        tcx.mir_keys(()).iter().map(|def_id| def_id.to_def_id()).collect()
    }
}

///////////////////////////////////////////////////////////////////////////
// Basic blocks and their parts (statements, terminators, ...)

/// Write out a human-readable textual representation for the given basic block.
fn write_basic_block<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    block: BasicBlock,
    body: &Body<'tcx>,
    extra_data: &mut F,
    w: &mut dyn io::Write,
    options: PrettyPrintMirOptions,
) -> io::Result<()>
where
    F: FnMut(PassWhere, &mut dyn io::Write) -> io::Result<()>,
{
    let data = &body[block];

    // Basic block label at the top.
    let cleanup_text = if data.is_cleanup { " (cleanup)" } else { "" };
    writeln!(w, "{INDENT}{block:?}{cleanup_text}: {{")?;

    // List of statements in the middle.
    let mut current_location = Location { block, statement_index: 0 };
    for statement in &data.statements {
        extra_data(PassWhere::BeforeLocation(current_location), w)?;
        let indented_body = format!("{INDENT}{INDENT}{statement:?};");
        if options.include_extra_comments {
            writeln!(
                w,
                "{:A$} // {}{}",
                indented_body,
                if tcx.sess.verbose_internals() {
                    format!("{current_location:?}: ")
                } else {
                    String::new()
                },
                comment(tcx, statement.source_info),
                A = ALIGN,
            )?;
        } else {
            writeln!(w, "{indented_body}")?;
        }

        write_extra(
            tcx,
            w,
            |visitor| {
                visitor.visit_statement(statement, current_location);
            },
            options,
        )?;

        extra_data(PassWhere::AfterLocation(current_location), w)?;

        current_location.statement_index += 1;
    }

    // Terminator at the bottom.
    extra_data(PassWhere::BeforeLocation(current_location), w)?;
    if data.terminator.is_some() {
        let indented_terminator = format!("{0}{0}{1:?};", INDENT, data.terminator().kind);
        if options.include_extra_comments {
            writeln!(
                w,
                "{:A$} // {}{}",
                indented_terminator,
                if tcx.sess.verbose_internals() {
                    format!("{current_location:?}: ")
                } else {
                    String::new()
                },
                comment(tcx, data.terminator().source_info),
                A = ALIGN,
            )?;
        } else {
            writeln!(w, "{indented_terminator}")?;
        }

        write_extra(
            tcx,
            w,
            |visitor| {
                visitor.visit_terminator(data.terminator(), current_location);
            },
            options,
        )?;
    }

    extra_data(PassWhere::AfterLocation(current_location), w)?;
    extra_data(PassWhere::AfterTerminator(block), w)?;

    writeln!(w, "{INDENT}}}")
}

impl Debug for Statement<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use self::StatementKind::*;
        match self.kind {
            Assign(box (ref place, ref rv)) => write!(fmt, "{place:?} = {rv:?}"),
            FakeRead(box (ref cause, ref place)) => {
                write!(fmt, "FakeRead({cause:?}, {place:?})")
            }
            Retag(ref kind, ref place) => write!(
                fmt,
                "Retag({}{:?})",
                match kind {
                    RetagKind::FnEntry => "[fn entry] ",
                    RetagKind::TwoPhase => "[2phase] ",
                    RetagKind::Raw => "[raw] ",
                    RetagKind::Default => "",
                },
                place,
            ),
            StorageLive(ref place) => write!(fmt, "StorageLive({place:?})"),
            StorageDead(ref place) => write!(fmt, "StorageDead({place:?})"),
            SetDiscriminant { ref place, variant_index } => {
                write!(fmt, "discriminant({place:?}) = {variant_index:?}")
            }
            Deinit(ref place) => write!(fmt, "Deinit({place:?})"),
            PlaceMention(ref place) => {
                write!(fmt, "PlaceMention({place:?})")
            }
            AscribeUserType(box (ref place, ref c_ty), ref variance) => {
                write!(fmt, "AscribeUserType({place:?}, {variance:?}, {c_ty:?})")
            }
            Coverage(ref kind) => write!(fmt, "Coverage::{kind:?}"),
            Intrinsic(box ref intrinsic) => write!(fmt, "{intrinsic}"),
            ConstEvalCounter => write!(fmt, "ConstEvalCounter"),
            Nop => write!(fmt, "nop"),
            BackwardIncompatibleDropHint { ref place, reason: _ } => {
                // For now, we don't record the reason because there is only one use case,
                // which is to report breaking change in drop order by Edition 2024
                write!(fmt, "BackwardIncompatibleDropHint({place:?})")
            }
        }
    }
}

impl Display for NonDivergingIntrinsic<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Assume(op) => write!(f, "assume({op:?})"),
            Self::CopyNonOverlapping(CopyNonOverlapping { src, dst, count }) => {
                write!(f, "copy_nonoverlapping(dst = {dst:?}, src = {src:?}, count = {count:?})")
            }
        }
    }
}

impl<'tcx> Debug for TerminatorKind<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_head(fmt)?;
        let successor_count = self.successors().count();
        let labels = self.fmt_successor_labels();
        assert_eq!(successor_count, labels.len());

        // `Cleanup` is already included in successors
        let show_unwind = !matches!(self.unwind(), None | Some(UnwindAction::Cleanup(_)));
        let fmt_unwind = |fmt: &mut Formatter<'_>| -> fmt::Result {
            write!(fmt, "unwind ")?;
            match self.unwind() {
                // Not needed or included in successors
                None | Some(UnwindAction::Cleanup(_)) => unreachable!(),
                Some(UnwindAction::Continue) => write!(fmt, "continue"),
                Some(UnwindAction::Unreachable) => write!(fmt, "unreachable"),
                Some(UnwindAction::Terminate(reason)) => {
                    write!(fmt, "terminate({})", reason.as_short_str())
                }
            }
        };

        match (successor_count, show_unwind) {
            (0, false) => Ok(()),
            (0, true) => {
                write!(fmt, " -> ")?;
                fmt_unwind(fmt)
            }
            (1, false) => write!(fmt, " -> {:?}", self.successors().next().unwrap()),
            _ => {
                write!(fmt, " -> [")?;
                for (i, target) in self.successors().enumerate() {
                    if i > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{}: {:?}", labels[i], target)?;
                }
                if show_unwind {
                    write!(fmt, ", ")?;
                    fmt_unwind(fmt)?;
                }
                write!(fmt, "]")
            }
        }
    }
}

impl<'tcx> TerminatorKind<'tcx> {
    /// Writes the "head" part of the terminator; that is, its name and the data it uses to pick the
    /// successor basic block, if any. The only information not included is the list of possible
    /// successors, which may be rendered differently between the text and the graphviz format.
    pub fn fmt_head<W: fmt::Write>(&self, fmt: &mut W) -> fmt::Result {
        use self::TerminatorKind::*;
        match self {
            Goto { .. } => write!(fmt, "goto"),
            SwitchInt { discr, .. } => write!(fmt, "switchInt({discr:?})"),
            Return => write!(fmt, "return"),
            CoroutineDrop => write!(fmt, "coroutine_drop"),
            UnwindResume => write!(fmt, "resume"),
            UnwindTerminate(reason) => {
                write!(fmt, "terminate({})", reason.as_short_str())
            }
            Yield { value, resume_arg, .. } => write!(fmt, "{resume_arg:?} = yield({value:?})"),
            Unreachable => write!(fmt, "unreachable"),
            Drop { place, .. } => write!(fmt, "drop({place:?})"),
            Call { func, args, destination, .. } => {
                write!(fmt, "{destination:?} = ")?;
                write!(fmt, "{func:?}(")?;
                for (index, arg) in args.iter().map(|a| &a.node).enumerate() {
                    if index > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{arg:?}")?;
                }
                write!(fmt, ")")
            }
            TailCall { func, args, .. } => {
                write!(fmt, "tailcall {func:?}(")?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{:?}", arg)?;
                }
                write!(fmt, ")")
            }
            Assert { cond, expected, msg, .. } => {
                write!(fmt, "assert(")?;
                if !expected {
                    write!(fmt, "!")?;
                }
                write!(fmt, "{cond:?}, ")?;
                msg.fmt_assert_args(fmt)?;
                write!(fmt, ")")
            }
            FalseEdge { .. } => write!(fmt, "falseEdge"),
            FalseUnwind { .. } => write!(fmt, "falseUnwind"),
            InlineAsm { template, operands, options, .. } => {
                write!(fmt, "asm!(\"{}\"", InlineAsmTemplatePiece::to_string(template))?;
                for op in operands {
                    write!(fmt, ", ")?;
                    let print_late = |&late| if late { "late" } else { "" };
                    match op {
                        InlineAsmOperand::In { reg, value } => {
                            write!(fmt, "in({reg}) {value:?}")?;
                        }
                        InlineAsmOperand::Out { reg, late, place: Some(place) } => {
                            write!(fmt, "{}out({}) {:?}", print_late(late), reg, place)?;
                        }
                        InlineAsmOperand::Out { reg, late, place: None } => {
                            write!(fmt, "{}out({}) _", print_late(late), reg)?;
                        }
                        InlineAsmOperand::InOut {
                            reg,
                            late,
                            in_value,
                            out_place: Some(out_place),
                        } => {
                            write!(
                                fmt,
                                "in{}out({}) {:?} => {:?}",
                                print_late(late),
                                reg,
                                in_value,
                                out_place
                            )?;
                        }
                        InlineAsmOperand::InOut { reg, late, in_value, out_place: None } => {
                            write!(fmt, "in{}out({}) {:?} => _", print_late(late), reg, in_value)?;
                        }
                        InlineAsmOperand::Const { value } => {
                            write!(fmt, "const {value:?}")?;
                        }
                        InlineAsmOperand::SymFn { value } => {
                            write!(fmt, "sym_fn {value:?}")?;
                        }
                        InlineAsmOperand::SymStatic { def_id } => {
                            write!(fmt, "sym_static {def_id:?}")?;
                        }
                        InlineAsmOperand::Label { target_index } => {
                            write!(fmt, "label {target_index}")?;
                        }
                    }
                }
                write!(fmt, ", options({options:?}))")
            }
        }
    }

    /// Returns the list of labels for the edges to the successor basic blocks.
    pub fn fmt_successor_labels(&self) -> Vec<Cow<'static, str>> {
        use self::TerminatorKind::*;
        match *self {
            Return
            | TailCall { .. }
            | UnwindResume
            | UnwindTerminate(_)
            | Unreachable
            | CoroutineDrop => vec![],
            Goto { .. } => vec!["".into()],
            SwitchInt { ref targets, .. } => targets
                .values
                .iter()
                .map(|&u| Cow::Owned(u.to_string()))
                .chain(iter::once("otherwise".into()))
                .collect(),
            Call { target: Some(_), unwind: UnwindAction::Cleanup(_), .. } => {
                vec!["return".into(), "unwind".into()]
            }
            Call { target: Some(_), unwind: _, .. } => vec!["return".into()],
            Call { target: None, unwind: UnwindAction::Cleanup(_), .. } => vec!["unwind".into()],
            Call { target: None, unwind: _, .. } => vec![],
            Yield { drop: Some(_), .. } => vec!["resume".into(), "drop".into()],
            Yield { drop: None, .. } => vec!["resume".into()],
            Drop { unwind: UnwindAction::Cleanup(_), .. } => vec!["return".into(), "unwind".into()],
            Drop { unwind: _, .. } => vec!["return".into()],
            Assert { unwind: UnwindAction::Cleanup(_), .. } => {
                vec!["success".into(), "unwind".into()]
            }
            Assert { unwind: _, .. } => vec!["success".into()],
            FalseEdge { .. } => vec!["real".into(), "imaginary".into()],
            FalseUnwind { unwind: UnwindAction::Cleanup(_), .. } => {
                vec!["real".into(), "unwind".into()]
            }
            FalseUnwind { unwind: _, .. } => vec!["real".into()],
            InlineAsm { asm_macro, options, ref targets, unwind, .. } => {
                let mut vec = Vec::with_capacity(targets.len() + 1);
                if !asm_macro.diverges(options) {
                    vec.push("return".into());
                }
                vec.resize(targets.len(), "label".into());

                if let UnwindAction::Cleanup(_) = unwind {
                    vec.push("unwind".into());
                }

                vec
            }
        }
    }
}

impl<'tcx> Debug for Rvalue<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use self::Rvalue::*;

        match *self {
            Use(ref place) => write!(fmt, "{place:?}"),
            Repeat(ref a, b) => {
                write!(fmt, "[{a:?}; ")?;
                pretty_print_const(b, fmt, false)?;
                write!(fmt, "]")
            }
            Len(ref a) => write!(fmt, "Len({a:?})"),
            Cast(ref kind, ref place, ref ty) => {
                with_no_trimmed_paths!(write!(fmt, "{place:?} as {ty} ({kind:?})"))
            }
            BinaryOp(ref op, box (ref a, ref b)) => write!(fmt, "{op:?}({a:?}, {b:?})"),
            UnaryOp(ref op, ref a) => write!(fmt, "{op:?}({a:?})"),
            Discriminant(ref place) => write!(fmt, "discriminant({place:?})"),
            NullaryOp(ref op, ref t) => {
                let t = with_no_trimmed_paths!(format!("{}", t));
                match op {
                    NullOp::SizeOf => write!(fmt, "SizeOf({t})"),
                    NullOp::AlignOf => write!(fmt, "AlignOf({t})"),
                    NullOp::OffsetOf(fields) => write!(fmt, "OffsetOf({t}, {fields:?})"),
                    NullOp::UbChecks => write!(fmt, "UbChecks()"),
                    NullOp::ContractChecks => write!(fmt, "ContractChecks()"),
                }
            }
            ThreadLocalRef(did) => ty::tls::with(|tcx| {
                let muta = tcx.static_mutability(did).unwrap().prefix_str();
                write!(fmt, "&/*tls*/ {}{}", muta, tcx.def_path_str(did))
            }),
            Ref(region, borrow_kind, ref place) => {
                let kind_str = match borrow_kind {
                    BorrowKind::Shared => "",
                    BorrowKind::Fake(FakeBorrowKind::Deep) => "fake ",
                    BorrowKind::Fake(FakeBorrowKind::Shallow) => "fake shallow ",
                    BorrowKind::Mut { .. } => "mut ",
                };

                // When printing regions, add trailing space if necessary.
                let print_region = ty::tls::with(|tcx| {
                    tcx.sess.verbose_internals() || tcx.sess.opts.unstable_opts.identify_regions
                });
                let region = if print_region {
                    let mut region = region.to_string();
                    if !region.is_empty() {
                        region.push(' ');
                    }
                    region
                } else {
                    // Do not even print 'static
                    String::new()
                };
                write!(fmt, "&{region}{kind_str}{place:?}")
            }

            CopyForDeref(ref place) => write!(fmt, "deref_copy {place:#?}"),

            RawPtr(mutability, ref place) => {
                write!(fmt, "&raw {mut_str} {place:?}", mut_str = mutability.ptr_str())
            }

            Aggregate(ref kind, ref places) => {
                let fmt_tuple = |fmt: &mut Formatter<'_>, name: &str| {
                    let mut tuple_fmt = fmt.debug_tuple(name);
                    for place in places {
                        tuple_fmt.field(place);
                    }
                    tuple_fmt.finish()
                };

                match **kind {
                    AggregateKind::Array(_) => write!(fmt, "{places:?}"),

                    AggregateKind::Tuple => {
                        if places.is_empty() {
                            write!(fmt, "()")
                        } else {
                            fmt_tuple(fmt, "")
                        }
                    }

                    AggregateKind::Adt(adt_did, variant, args, _user_ty, _) => {
                        ty::tls::with(|tcx| {
                            let variant_def = &tcx.adt_def(adt_did).variant(variant);
                            let args = tcx.lift(args).expect("could not lift for printing");
                            let name = FmtPrinter::print_string(tcx, Namespace::ValueNS, |cx| {
                                cx.print_def_path(variant_def.def_id, args)
                            })?;

                            match variant_def.ctor_kind() {
                                Some(CtorKind::Const) => fmt.write_str(&name),
                                Some(CtorKind::Fn) => fmt_tuple(fmt, &name),
                                None => {
                                    let mut struct_fmt = fmt.debug_struct(&name);
                                    for (field, place) in iter::zip(&variant_def.fields, places) {
                                        struct_fmt.field(field.name.as_str(), place);
                                    }
                                    struct_fmt.finish()
                                }
                            }
                        })
                    }

                    AggregateKind::Closure(def_id, args)
                    | AggregateKind::CoroutineClosure(def_id, args) => ty::tls::with(|tcx| {
                        let name = if tcx.sess.opts.unstable_opts.span_free_formats {
                            let args = tcx.lift(args).unwrap();
                            format!("{{closure@{}}}", tcx.def_path_str_with_args(def_id, args),)
                        } else {
                            let span = tcx.def_span(def_id);
                            format!(
                                "{{closure@{}}}",
                                tcx.sess.source_map().span_to_diagnostic_string(span)
                            )
                        };
                        let mut struct_fmt = fmt.debug_struct(&name);

                        // FIXME(project-rfc-2229#48): This should be a list of capture names/places
                        if let Some(def_id) = def_id.as_local()
                            && let Some(upvars) = tcx.upvars_mentioned(def_id)
                        {
                            for (&var_id, place) in iter::zip(upvars.keys(), places) {
                                let var_name = tcx.hir_name(var_id);
                                struct_fmt.field(var_name.as_str(), place);
                            }
                        } else {
                            for (index, place) in places.iter().enumerate() {
                                struct_fmt.field(&format!("{index}"), place);
                            }
                        }

                        struct_fmt.finish()
                    }),

                    AggregateKind::Coroutine(def_id, _) => ty::tls::with(|tcx| {
                        let name = format!("{{coroutine@{:?}}}", tcx.def_span(def_id));
                        let mut struct_fmt = fmt.debug_struct(&name);

                        // FIXME(project-rfc-2229#48): This should be a list of capture names/places
                        if let Some(def_id) = def_id.as_local()
                            && let Some(upvars) = tcx.upvars_mentioned(def_id)
                        {
                            for (&var_id, place) in iter::zip(upvars.keys(), places) {
                                let var_name = tcx.hir_name(var_id);
                                struct_fmt.field(var_name.as_str(), place);
                            }
                        } else {
                            for (index, place) in places.iter().enumerate() {
                                struct_fmt.field(&format!("{index}"), place);
                            }
                        }

                        struct_fmt.finish()
                    }),

                    AggregateKind::RawPtr(pointee_ty, mutability) => {
                        let kind_str = match mutability {
                            Mutability::Mut => "mut",
                            Mutability::Not => "const",
                        };
                        with_no_trimmed_paths!(write!(fmt, "*{kind_str} {pointee_ty} from "))?;
                        fmt_tuple(fmt, "")
                    }
                }
            }

            ShallowInitBox(ref place, ref ty) => {
                with_no_trimmed_paths!(write!(fmt, "ShallowInitBox({place:?}, {ty})"))
            }

            WrapUnsafeBinder(ref op, ty) => {
                with_no_trimmed_paths!(write!(fmt, "wrap_binder!({op:?}; {ty})"))
            }
        }
    }
}

impl<'tcx> Debug for Operand<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use self::Operand::*;
        match *self {
            Constant(ref a) => write!(fmt, "{a:?}"),
            Copy(ref place) => write!(fmt, "copy {place:?}"),
            Move(ref place) => write!(fmt, "move {place:?}"),
        }
    }
}

impl<'tcx> Debug for ConstOperand<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        write!(fmt, "{self}")
    }
}

impl<'tcx> Display for ConstOperand<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match self.ty().kind() {
            ty::FnDef(..) => {}
            _ => write!(fmt, "const ")?,
        }
        Display::fmt(&self.const_, fmt)
    }
}

impl Debug for Place<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        self.as_ref().fmt(fmt)
    }
}

impl Debug for PlaceRef<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        pre_fmt_projection(self.projection, fmt)?;
        write!(fmt, "{:?}", self.local)?;
        post_fmt_projection(self.projection, fmt)
    }
}

fn pre_fmt_projection(projection: &[PlaceElem<'_>], fmt: &mut Formatter<'_>) -> fmt::Result {
    for &elem in projection.iter().rev() {
        match elem {
            ProjectionElem::OpaqueCast(_)
            | ProjectionElem::Subtype(_)
            | ProjectionElem::Downcast(_, _)
            | ProjectionElem::Field(_, _) => {
                write!(fmt, "(")?;
            }
            ProjectionElem::Deref => {
                write!(fmt, "(*")?;
            }
            ProjectionElem::Index(_)
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. } => {}
            ProjectionElem::UnwrapUnsafeBinder(_) => {
                write!(fmt, "unwrap_binder!(")?;
            }
        }
    }

    Ok(())
}

fn post_fmt_projection(projection: &[PlaceElem<'_>], fmt: &mut Formatter<'_>) -> fmt::Result {
    for &elem in projection.iter() {
        match elem {
            ProjectionElem::OpaqueCast(ty) => {
                write!(fmt, " as {ty})")?;
            }
            ProjectionElem::Subtype(ty) => {
                write!(fmt, " as subtype {ty})")?;
            }
            ProjectionElem::Downcast(Some(name), _index) => {
                write!(fmt, " as {name})")?;
            }
            ProjectionElem::Downcast(None, index) => {
                write!(fmt, " as variant#{index:?})")?;
            }
            ProjectionElem::Deref => {
                write!(fmt, ")")?;
            }
            ProjectionElem::Field(field, ty) => {
                with_no_trimmed_paths!(write!(fmt, ".{:?}: {})", field.index(), ty)?);
            }
            ProjectionElem::Index(ref index) => {
                write!(fmt, "[{index:?}]")?;
            }
            ProjectionElem::ConstantIndex { offset, min_length, from_end: false } => {
                write!(fmt, "[{offset:?} of {min_length:?}]")?;
            }
            ProjectionElem::ConstantIndex { offset, min_length, from_end: true } => {
                write!(fmt, "[-{offset:?} of {min_length:?}]")?;
            }
            ProjectionElem::Subslice { from, to: 0, from_end: true } => {
                write!(fmt, "[{from:?}:]")?;
            }
            ProjectionElem::Subslice { from: 0, to, from_end: true } => {
                write!(fmt, "[:-{to:?}]")?;
            }
            ProjectionElem::Subslice { from, to, from_end: true } => {
                write!(fmt, "[{from:?}:-{to:?}]")?;
            }
            ProjectionElem::Subslice { from, to, from_end: false } => {
                write!(fmt, "[{from:?}..{to:?}]")?;
            }
            ProjectionElem::UnwrapUnsafeBinder(ty) => {
                write!(fmt, "; {ty})")?;
            }
        }
    }

    Ok(())
}

/// After we print the main statement, we sometimes dump extra
/// information. There's often a lot of little things "nuzzled up" in
/// a statement.
fn write_extra<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    write: &mut dyn io::Write,
    mut visit_op: F,
    options: PrettyPrintMirOptions,
) -> io::Result<()>
where
    F: FnMut(&mut ExtraComments<'tcx>),
{
    if options.include_extra_comments {
        let mut extra_comments = ExtraComments { tcx, comments: vec![] };
        visit_op(&mut extra_comments);
        for comment in extra_comments.comments {
            writeln!(write, "{:A$} // {}", "", comment, A = ALIGN)?;
        }
    }
    Ok(())
}

struct ExtraComments<'tcx> {
    tcx: TyCtxt<'tcx>,
    comments: Vec<String>,
}

impl<'tcx> ExtraComments<'tcx> {
    fn push(&mut self, lines: &str) {
        for line in lines.split('\n') {
            self.comments.push(line.to_string());
        }
    }
}

fn use_verbose(ty: Ty<'_>, fn_def: bool) -> bool {
    match *ty.kind() {
        ty::Int(_) | ty::Uint(_) | ty::Bool | ty::Char | ty::Float(_) => false,
        // Unit type
        ty::Tuple(g_args) if g_args.is_empty() => false,
        ty::Tuple(g_args) => g_args.iter().any(|g_arg| use_verbose(g_arg, fn_def)),
        ty::Array(ty, _) => use_verbose(ty, fn_def),
        ty::FnDef(..) => fn_def,
        _ => true,
    }
}

impl<'tcx> Visitor<'tcx> for ExtraComments<'tcx> {
    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, _location: Location) {
        let ConstOperand { span, user_ty, const_ } = constant;
        if use_verbose(const_.ty(), true) {
            self.push("mir::ConstOperand");
            self.push(&format!(
                "+ span: {}",
                self.tcx.sess.source_map().span_to_embeddable_string(*span)
            ));
            if let Some(user_ty) = user_ty {
                self.push(&format!("+ user_ty: {user_ty:?}"));
            }

            let fmt_val = |val: ConstValue<'tcx>, ty: Ty<'tcx>| {
                let tcx = self.tcx;
                rustc_data_structures::make_display(move |fmt| {
                    pretty_print_const_value_tcx(tcx, val, ty, fmt)
                })
            };

            let fmt_valtree = |cv: &ty::Value<'tcx>| {
                let mut cx = FmtPrinter::new(self.tcx, Namespace::ValueNS);
                cx.pretty_print_const_valtree(*cv, /*print_ty*/ true).unwrap();
                cx.into_buffer()
            };

            let val = match const_ {
                Const::Ty(_, ct) => match ct.kind() {
                    ty::ConstKind::Param(p) => format!("ty::Param({p})"),
                    ty::ConstKind::Unevaluated(uv) => {
                        format!("ty::Unevaluated({}, {:?})", self.tcx.def_path_str(uv.def), uv.args,)
                    }
                    ty::ConstKind::Value(cv) => {
                        format!("ty::Valtree({})", fmt_valtree(&cv))
                    }
                    // No `ty::` prefix since we also use this to represent errors from `mir::Unevaluated`.
                    ty::ConstKind::Error(_) => "Error".to_string(),
                    // These variants shouldn't exist in the MIR.
                    ty::ConstKind::Placeholder(_)
                    | ty::ConstKind::Infer(_)
                    | ty::ConstKind::Expr(_)
                    | ty::ConstKind::Bound(..) => bug!("unexpected MIR constant: {:?}", const_),
                },
                Const::Unevaluated(uv, _) => {
                    format!(
                        "Unevaluated({}, {:?}, {:?})",
                        self.tcx.def_path_str(uv.def),
                        uv.args,
                        uv.promoted,
                    )
                }
                Const::Val(val, ty) => format!("Value({})", fmt_val(*val, *ty)),
            };

            // This reflects what `Const` looked liked before `val` was renamed
            // as `kind`. We print it like this to avoid having to update
            // expected output in a lot of tests.
            self.push(&format!("+ const_: Const {{ ty: {}, val: {} }}", const_.ty(), val));
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        if let Rvalue::Aggregate(kind, _) = rvalue {
            match **kind {
                AggregateKind::Closure(def_id, args) => {
                    self.push("closure");
                    self.push(&format!("+ def_id: {def_id:?}"));
                    self.push(&format!("+ args: {args:#?}"));
                }

                AggregateKind::Coroutine(def_id, args) => {
                    self.push("coroutine");
                    self.push(&format!("+ def_id: {def_id:?}"));
                    self.push(&format!("+ args: {args:#?}"));
                    self.push(&format!("+ kind: {:?}", self.tcx.coroutine_kind(def_id)));
                }

                AggregateKind::Adt(_, _, _, Some(user_ty), _) => {
                    self.push("adt");
                    self.push(&format!("+ user_ty: {user_ty:?}"));
                }

                _ => {}
            }
        }
    }
}

fn comment(tcx: TyCtxt<'_>, SourceInfo { span, scope }: SourceInfo) -> String {
    let location = tcx.sess.source_map().span_to_embeddable_string(span);
    format!("scope {} at {}", scope.index(), location,)
}

///////////////////////////////////////////////////////////////////////////
// Allocations

/// Find all `AllocId`s mentioned (recursively) in the MIR body and print their corresponding
/// allocations.
pub fn write_allocations<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'_>,
    w: &mut dyn io::Write,
) -> io::Result<()> {
    fn alloc_ids_from_alloc(
        alloc: ConstAllocation<'_>,
    ) -> impl DoubleEndedIterator<Item = AllocId> {
        alloc.inner().provenance().ptrs().values().map(|p| p.alloc_id())
    }

    fn alloc_id_from_const_val(val: ConstValue<'_>) -> Option<AllocId> {
        match val {
            ConstValue::Scalar(interpret::Scalar::Ptr(ptr, _)) => Some(ptr.provenance.alloc_id()),
            ConstValue::Scalar(interpret::Scalar::Int { .. }) => None,
            ConstValue::ZeroSized => None,
            ConstValue::Slice { .. } => {
                // `u8`/`str` slices, shouldn't contain pointers that we want to print.
                None
            }
            ConstValue::Indirect { alloc_id, .. } => {
                // FIXME: we don't actually want to print all of these, since some are printed nicely directly as values inline in MIR.
                // Really we'd want `pretty_print_const_value` to decide which allocations to print, instead of having a separate visitor.
                Some(alloc_id)
            }
        }
    }
    struct CollectAllocIds(BTreeSet<AllocId>);

    impl<'tcx> Visitor<'tcx> for CollectAllocIds {
        fn visit_const_operand(&mut self, c: &ConstOperand<'tcx>, _: Location) {
            match c.const_ {
                Const::Ty(_, _) | Const::Unevaluated(..) => {}
                Const::Val(val, _) => {
                    if let Some(id) = alloc_id_from_const_val(val) {
                        self.0.insert(id);
                    }
                }
            }
        }
    }

    let mut visitor = CollectAllocIds(Default::default());
    visitor.visit_body(body);

    // `seen` contains all seen allocations, including the ones we have *not* printed yet.
    // The protocol is to first `insert` into `seen`, and only if that returns `true`
    // then push to `todo`.
    let mut seen = visitor.0;
    let mut todo: Vec<_> = seen.iter().copied().collect();
    while let Some(id) = todo.pop() {
        let mut write_allocation_track_relocs =
            |w: &mut dyn io::Write, alloc: ConstAllocation<'tcx>| -> io::Result<()> {
                // `.rev()` because we are popping them from the back of the `todo` vector.
                for id in alloc_ids_from_alloc(alloc).rev() {
                    if seen.insert(id) {
                        todo.push(id);
                    }
                }
                write!(w, "{}", display_allocation(tcx, alloc.inner()))
            };
        write!(w, "\n{id:?}")?;
        match tcx.try_get_global_alloc(id) {
            // This can't really happen unless there are bugs, but it doesn't cost us anything to
            // gracefully handle it and allow buggy rustc to be debugged via allocation printing.
            None => write!(w, " (deallocated)")?,
            Some(GlobalAlloc::Function { instance, .. }) => write!(w, " (fn: {instance})")?,
            Some(GlobalAlloc::VTable(ty, dyn_ty)) => {
                write!(w, " (vtable: impl {dyn_ty} for {ty})")?
            }
            Some(GlobalAlloc::Static(did)) if !tcx.is_foreign_item(did) => {
                write!(w, " (static: {}", tcx.def_path_str(did))?;
                if body.phase <= MirPhase::Runtime(RuntimePhase::PostCleanup)
                    && tcx.hir_body_const_context(body.source.def_id()).is_some()
                {
                    // Statics may be cyclic and evaluating them too early
                    // in the MIR pipeline may cause cycle errors even though
                    // normal compilation is fine.
                    write!(w, ")")?;
                } else {
                    match tcx.eval_static_initializer(did) {
                        Ok(alloc) => {
                            write!(w, ", ")?;
                            write_allocation_track_relocs(w, alloc)?;
                        }
                        Err(_) => write!(w, ", error during initializer evaluation)")?,
                    }
                }
            }
            Some(GlobalAlloc::Static(did)) => {
                write!(w, " (extern static: {})", tcx.def_path_str(did))?
            }
            Some(GlobalAlloc::Memory(alloc)) => {
                write!(w, " (")?;
                write_allocation_track_relocs(w, alloc)?
            }
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Dumps the size and metadata and content of an allocation to the given writer.
/// The expectation is that the caller first prints other relevant metadata, so the exact
/// format of this function is (*without* leading or trailing newline):
///
/// ```text
/// size: {}, align: {}) {
///     <bytes>
/// }
/// ```
///
/// The byte format is similar to how hex editors print bytes. Each line starts with the address of
/// the start of the line, followed by all bytes in hex format (space separated).
/// If the allocation is small enough to fit into a single line, no start address is given.
/// After the hex dump, an ascii dump follows, replacing all unprintable characters (control
/// characters or characters whose value is larger than 127) with a `.`
/// This also prints provenance adequately.
pub fn display_allocation<'a, 'tcx, Prov: Provenance, Extra, Bytes: AllocBytes>(
    tcx: TyCtxt<'tcx>,
    alloc: &'a Allocation<Prov, Extra, Bytes>,
) -> RenderAllocation<'a, 'tcx, Prov, Extra, Bytes> {
    RenderAllocation { tcx, alloc }
}

#[doc(hidden)]
pub struct RenderAllocation<'a, 'tcx, Prov: Provenance, Extra, Bytes: AllocBytes> {
    tcx: TyCtxt<'tcx>,
    alloc: &'a Allocation<Prov, Extra, Bytes>,
}

impl<'a, 'tcx, Prov: Provenance, Extra, Bytes: AllocBytes> std::fmt::Display
    for RenderAllocation<'a, 'tcx, Prov, Extra, Bytes>
{
    fn fmt(&self, w: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let RenderAllocation { tcx, alloc } = *self;
        write!(w, "size: {}, align: {})", alloc.size().bytes(), alloc.align.bytes())?;
        if alloc.size() == Size::ZERO {
            // We are done.
            return write!(w, " {{}}");
        }
        if tcx.sess.opts.unstable_opts.dump_mir_exclude_alloc_bytes {
            return write!(w, " {{ .. }}");
        }
        // Write allocation bytes.
        writeln!(w, " {{")?;
        write_allocation_bytes(tcx, alloc, w, "    ")?;
        write!(w, "}}")?;
        Ok(())
    }
}

fn write_allocation_endline(w: &mut dyn std::fmt::Write, ascii: &str) -> std::fmt::Result {
    for _ in 0..(BYTES_PER_LINE - ascii.chars().count()) {
        write!(w, "   ")?;
    }
    writeln!(w, " │ {ascii}")
}

/// Number of bytes to print per allocation hex dump line.
const BYTES_PER_LINE: usize = 16;

/// Prints the line start address and returns the new line start address.
fn write_allocation_newline(
    w: &mut dyn std::fmt::Write,
    mut line_start: Size,
    ascii: &str,
    pos_width: usize,
    prefix: &str,
) -> Result<Size, std::fmt::Error> {
    write_allocation_endline(w, ascii)?;
    line_start += Size::from_bytes(BYTES_PER_LINE);
    write!(w, "{}0x{:02$x} │ ", prefix, line_start.bytes(), pos_width)?;
    Ok(line_start)
}

/// The `prefix` argument allows callers to add an arbitrary prefix before each line (even if there
/// is only one line). Note that your prefix should contain a trailing space as the lines are
/// printed directly after it.
pub fn write_allocation_bytes<'tcx, Prov: Provenance, Extra, Bytes: AllocBytes>(
    tcx: TyCtxt<'tcx>,
    alloc: &Allocation<Prov, Extra, Bytes>,
    w: &mut dyn std::fmt::Write,
    prefix: &str,
) -> std::fmt::Result {
    let num_lines = alloc.size().bytes_usize().saturating_sub(BYTES_PER_LINE);
    // Number of chars needed to represent all line numbers.
    let pos_width = hex_number_length(alloc.size().bytes());

    if num_lines > 0 {
        write!(w, "{}0x{:02$x} │ ", prefix, 0, pos_width)?;
    } else {
        write!(w, "{prefix}")?;
    }

    let mut i = Size::ZERO;
    let mut line_start = Size::ZERO;

    let ptr_size = tcx.data_layout.pointer_size;

    let mut ascii = String::new();

    let oversized_ptr = |target: &mut String, width| {
        if target.len() > width {
            write!(target, " ({} ptr bytes)", ptr_size.bytes()).unwrap();
        }
    };

    while i < alloc.size() {
        // The line start already has a space. While we could remove that space from the line start
        // printing and unconditionally print a space here, that would cause the single-line case
        // to have a single space before it, which looks weird.
        if i != line_start {
            write!(w, " ")?;
        }
        if let Some(prov) = alloc.provenance().get_ptr(i) {
            // Memory with provenance must be defined
            assert!(alloc.init_mask().is_range_initialized(alloc_range(i, ptr_size)).is_ok());
            let j = i.bytes_usize();
            let offset = alloc
                .inspect_with_uninit_and_ptr_outside_interpreter(j..j + ptr_size.bytes_usize());
            let offset = read_target_uint(tcx.data_layout.endian, offset).unwrap();
            let offset = Size::from_bytes(offset);
            let provenance_width = |bytes| bytes * 3;
            let ptr = Pointer::new(prov, offset);
            let mut target = format!("{ptr:?}");
            if target.len() > provenance_width(ptr_size.bytes_usize() - 1) {
                // This is too long, try to save some space.
                target = format!("{ptr:#?}");
            }
            if ((i - line_start) + ptr_size).bytes_usize() > BYTES_PER_LINE {
                // This branch handles the situation where a provenance starts in the current line
                // but ends in the next one.
                let remainder = Size::from_bytes(BYTES_PER_LINE) - (i - line_start);
                let overflow = ptr_size - remainder;
                let remainder_width = provenance_width(remainder.bytes_usize()) - 2;
                let overflow_width = provenance_width(overflow.bytes_usize() - 1) + 1;
                ascii.push('╾'); // HEAVY LEFT AND LIGHT RIGHT
                for _ in 1..remainder.bytes() {
                    ascii.push('─'); // LIGHT HORIZONTAL
                }
                if overflow_width > remainder_width && overflow_width >= target.len() {
                    // The case where the provenance fits into the part in the next line
                    write!(w, "╾{0:─^1$}", "", remainder_width)?;
                    line_start =
                        write_allocation_newline(w, line_start, &ascii, pos_width, prefix)?;
                    ascii.clear();
                    write!(w, "{target:─^overflow_width$}╼")?;
                } else {
                    oversized_ptr(&mut target, remainder_width);
                    write!(w, "╾{target:─^remainder_width$}")?;
                    line_start =
                        write_allocation_newline(w, line_start, &ascii, pos_width, prefix)?;
                    write!(w, "{0:─^1$}╼", "", overflow_width)?;
                    ascii.clear();
                }
                for _ in 0..overflow.bytes() - 1 {
                    ascii.push('─');
                }
                ascii.push('╼'); // LIGHT LEFT AND HEAVY RIGHT
                i += ptr_size;
                continue;
            } else {
                // This branch handles a provenance that starts and ends in the current line.
                let provenance_width = provenance_width(ptr_size.bytes_usize() - 1);
                oversized_ptr(&mut target, provenance_width);
                ascii.push('╾');
                write!(w, "╾{target:─^provenance_width$}╼")?;
                for _ in 0..ptr_size.bytes() - 2 {
                    ascii.push('─');
                }
                ascii.push('╼');
                i += ptr_size;
            }
        } else if let Some(prov) = alloc.provenance().get(i, &tcx) {
            // Memory with provenance must be defined
            assert!(
                alloc.init_mask().is_range_initialized(alloc_range(i, Size::from_bytes(1))).is_ok()
            );
            ascii.push('━'); // HEAVY HORIZONTAL
            // We have two characters to display this, which is obviously not enough.
            // Format is similar to "oversized" above.
            let j = i.bytes_usize();
            let c = alloc.inspect_with_uninit_and_ptr_outside_interpreter(j..j + 1)[0];
            write!(w, "╾{c:02x}{prov:#?} (1 ptr byte)╼")?;
            i += Size::from_bytes(1);
        } else if alloc
            .init_mask()
            .is_range_initialized(alloc_range(i, Size::from_bytes(1)))
            .is_ok()
        {
            let j = i.bytes_usize();

            // Checked definedness (and thus range) and provenance. This access also doesn't
            // influence interpreter execution but is only for debugging.
            let c = alloc.inspect_with_uninit_and_ptr_outside_interpreter(j..j + 1)[0];
            write!(w, "{c:02x}")?;
            if c.is_ascii_control() || c >= 0x80 {
                ascii.push('.');
            } else {
                ascii.push(char::from(c));
            }
            i += Size::from_bytes(1);
        } else {
            write!(w, "__")?;
            ascii.push('░');
            i += Size::from_bytes(1);
        }
        // Print a new line header if the next line still has some bytes to print.
        if i == line_start + Size::from_bytes(BYTES_PER_LINE) && i != alloc.size() {
            line_start = write_allocation_newline(w, line_start, &ascii, pos_width, prefix)?;
            ascii.clear();
        }
    }
    write_allocation_endline(w, &ascii)?;

    Ok(())
}

///////////////////////////////////////////////////////////////////////////
// Constants

fn pretty_print_byte_str(fmt: &mut Formatter<'_>, byte_str: &[u8]) -> fmt::Result {
    write!(fmt, "b\"{}\"", byte_str.escape_ascii())
}

fn comma_sep<'tcx>(
    tcx: TyCtxt<'tcx>,
    fmt: &mut Formatter<'_>,
    elems: Vec<(ConstValue<'tcx>, Ty<'tcx>)>,
) -> fmt::Result {
    let mut first = true;
    for (ct, ty) in elems {
        if !first {
            fmt.write_str(", ")?;
        }
        pretty_print_const_value_tcx(tcx, ct, ty, fmt)?;
        first = false;
    }
    Ok(())
}

fn pretty_print_const_value_tcx<'tcx>(
    tcx: TyCtxt<'tcx>,
    ct: ConstValue<'tcx>,
    ty: Ty<'tcx>,
    fmt: &mut Formatter<'_>,
) -> fmt::Result {
    use crate::ty::print::PrettyPrinter;

    if tcx.sess.verbose_internals() {
        fmt.write_str(&format!("ConstValue({ct:?}: {ty})"))?;
        return Ok(());
    }

    let u8_type = tcx.types.u8;
    match (ct, ty.kind()) {
        // Byte/string slices, printed as (byte) string literals.
        (_, ty::Ref(_, inner_ty, _)) if matches!(inner_ty.kind(), ty::Str) => {
            if let Some(data) = ct.try_get_slice_bytes_for_diagnostics(tcx) {
                fmt.write_str(&format!("{:?}", String::from_utf8_lossy(data)))?;
                return Ok(());
            }
        }
        (_, ty::Ref(_, inner_ty, _)) if matches!(inner_ty.kind(), ty::Slice(t) if *t == u8_type) => {
            if let Some(data) = ct.try_get_slice_bytes_for_diagnostics(tcx) {
                pretty_print_byte_str(fmt, data)?;
                return Ok(());
            }
        }
        (ConstValue::Indirect { alloc_id, offset }, ty::Array(t, n)) if *t == u8_type => {
            let n = n.try_to_target_usize(tcx).unwrap();
            let alloc = tcx.global_alloc(alloc_id).unwrap_memory();
            // cast is ok because we already checked for pointer size (32 or 64 bit) above
            let range = AllocRange { start: offset, size: Size::from_bytes(n) };
            let byte_str = alloc.inner().get_bytes_strip_provenance(&tcx, range).unwrap();
            fmt.write_str("*")?;
            pretty_print_byte_str(fmt, byte_str)?;
            return Ok(());
        }
        // Aggregates, printed as array/tuple/struct/variant construction syntax.
        //
        // NB: the `has_non_region_param` check ensures that we can use
        // the `destructure_const` query with an empty `ty::ParamEnv` without
        // introducing ICEs (e.g. via `layout_of`) from missing bounds.
        // E.g. `transmute([0usize; 2]): (u8, *mut T)` needs to know `T: Sized`
        // to be able to destructure the tuple into `(0u8, *mut T)`
        (_, ty::Array(..) | ty::Tuple(..) | ty::Adt(..)) if !ty.has_non_region_param() => {
            let ct = tcx.lift(ct).unwrap();
            let ty = tcx.lift(ty).unwrap();
            if let Some(contents) = tcx.try_destructure_mir_constant_for_user_output(ct, ty) {
                let fields: Vec<(ConstValue<'_>, Ty<'_>)> = contents.fields.to_vec();
                match *ty.kind() {
                    ty::Array(..) => {
                        fmt.write_str("[")?;
                        comma_sep(tcx, fmt, fields)?;
                        fmt.write_str("]")?;
                    }
                    ty::Tuple(..) => {
                        fmt.write_str("(")?;
                        comma_sep(tcx, fmt, fields)?;
                        if contents.fields.len() == 1 {
                            fmt.write_str(",")?;
                        }
                        fmt.write_str(")")?;
                    }
                    ty::Adt(def, _) if def.variants().is_empty() => {
                        fmt.write_str(&format!("{{unreachable(): {ty}}}"))?;
                    }
                    ty::Adt(def, args) => {
                        let variant_idx = contents
                            .variant
                            .expect("destructed mir constant of adt without variant idx");
                        let variant_def = &def.variant(variant_idx);
                        let args = tcx.lift(args).unwrap();
                        let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
                        cx.print_alloc_ids = true;
                        cx.print_value_path(variant_def.def_id, args)?;
                        fmt.write_str(&cx.into_buffer())?;

                        match variant_def.ctor_kind() {
                            Some(CtorKind::Const) => {}
                            Some(CtorKind::Fn) => {
                                fmt.write_str("(")?;
                                comma_sep(tcx, fmt, fields)?;
                                fmt.write_str(")")?;
                            }
                            None => {
                                fmt.write_str(" {{ ")?;
                                let mut first = true;
                                for (field_def, (ct, ty)) in iter::zip(&variant_def.fields, fields)
                                {
                                    if !first {
                                        fmt.write_str(", ")?;
                                    }
                                    write!(fmt, "{}: ", field_def.name)?;
                                    pretty_print_const_value_tcx(tcx, ct, ty, fmt)?;
                                    first = false;
                                }
                                fmt.write_str(" }}")?;
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                return Ok(());
            }
        }
        (ConstValue::Scalar(scalar), _) => {
            let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
            cx.print_alloc_ids = true;
            let ty = tcx.lift(ty).unwrap();
            cx.pretty_print_const_scalar(scalar, ty)?;
            fmt.write_str(&cx.into_buffer())?;
            return Ok(());
        }
        (ConstValue::ZeroSized, ty::FnDef(d, s)) => {
            let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
            cx.print_alloc_ids = true;
            cx.print_value_path(*d, s)?;
            fmt.write_str(&cx.into_buffer())?;
            return Ok(());
        }
        // FIXME(oli-obk): also pretty print arrays and other aggregate constants by reading
        // their fields instead of just dumping the memory.
        _ => {}
    }
    // Fall back to debug pretty printing for invalid constants.
    write!(fmt, "{ct:?}: {ty}")
}

pub(crate) fn pretty_print_const_value<'tcx>(
    ct: ConstValue<'tcx>,
    ty: Ty<'tcx>,
    fmt: &mut Formatter<'_>,
) -> fmt::Result {
    ty::tls::with(|tcx| {
        let ct = tcx.lift(ct).unwrap();
        let ty = tcx.lift(ty).unwrap();
        pretty_print_const_value_tcx(tcx, ct, ty, fmt)
    })
}

///////////////////////////////////////////////////////////////////////////
// Miscellaneous

/// Calc converted u64 decimal into hex and return its length in chars.
///
/// ```ignore (cannot-test-private-function)
/// assert_eq!(1, hex_number_length(0));
/// assert_eq!(1, hex_number_length(1));
/// assert_eq!(2, hex_number_length(16));
/// ```
fn hex_number_length(x: u64) -> usize {
    if x == 0 {
        return 1;
    }
    let mut length = 0;
    let mut x_left = x;
    while x_left > 0 {
        x_left /= 16;
        length += 1;
    }
    length
}
