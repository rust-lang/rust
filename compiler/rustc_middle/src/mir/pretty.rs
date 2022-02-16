use std::collections::BTreeSet;
use std::fmt::Display;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::graphviz::write_mir_fn_graphviz;
use super::spanview::write_mir_fn_spanview;
use either::Either;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_index::vec::Idx;
use rustc_middle::mir::interpret::{
    read_target_uint, AllocId, Allocation, ConstValue, GlobalAlloc, Pointer, Provenance,
};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::MirSource;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_target::abi::Size;

const INDENT: &str = "    ";
/// Alignment for lining up comments following MIR statements
pub(crate) const ALIGN: usize = 40;

/// An indication of where we are in the control flow graph. Used for printing
/// extra information in `dump_mir`
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

/// If the session is properly configured, dumps a human-readable
/// representation of the mir into:
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
    pass_num: Option<&dyn Display>,
    pass_name: &str,
    disambiguator: &dyn Display,
    body: &Body<'tcx>,
    extra_data: F,
) where
    F: FnMut(PassWhere, &mut dyn Write) -> io::Result<()>,
{
    if !dump_enabled(tcx, pass_name, body.source.def_id()) {
        return;
    }

    dump_matched_mir_node(tcx, pass_num, pass_name, disambiguator, body, extra_data);
}

pub fn dump_enabled<'tcx>(tcx: TyCtxt<'tcx>, pass_name: &str, def_id: DefId) -> bool {
    let filters = match tcx.sess.opts.debugging_opts.dump_mir {
        None => return false,
        Some(ref filters) => filters,
    };
    let node_path = ty::print::with_forced_impl_filename_line(|| {
        // see notes on #41697 below
        tcx.def_path_str(def_id)
    });
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

fn dump_matched_mir_node<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    pass_num: Option<&dyn Display>,
    pass_name: &str,
    disambiguator: &dyn Display,
    body: &Body<'tcx>,
    mut extra_data: F,
) where
    F: FnMut(PassWhere, &mut dyn Write) -> io::Result<()>,
{
    let _: io::Result<()> = try {
        let mut file =
            create_dump_file(tcx, "mir", pass_num, pass_name, disambiguator, body.source)?;
        let def_path = ty::print::with_forced_impl_filename_line(|| {
            // see notes on #41697 above
            tcx.def_path_str(body.source.def_id())
        });
        write!(file, "// MIR for `{}", def_path)?;
        match body.source.promoted {
            None => write!(file, "`")?,
            Some(promoted) => write!(file, "::{:?}`", promoted)?,
        }
        writeln!(file, " {} {}", disambiguator, pass_name)?;
        if let Some(ref layout) = body.generator_layout() {
            writeln!(file, "/* generator_layout = {:#?} */", layout)?;
        }
        writeln!(file)?;
        extra_data(PassWhere::BeforeCFG, &mut file)?;
        write_user_type_annotations(tcx, body, &mut file)?;
        write_mir_fn(tcx, body, &mut extra_data, &mut file)?;
        extra_data(PassWhere::AfterCFG, &mut file)?;
    };

    if tcx.sess.opts.debugging_opts.dump_mir_graphviz {
        let _: io::Result<()> = try {
            let mut file =
                create_dump_file(tcx, "dot", pass_num, pass_name, disambiguator, body.source)?;
            write_mir_fn_graphviz(tcx, body, false, &mut file)?;
        };
    }

    if let Some(spanview) = tcx.sess.opts.debugging_opts.dump_mir_spanview {
        let _: io::Result<()> = try {
            let file_basename =
                dump_file_basename(tcx, pass_num, pass_name, disambiguator, body.source);
            let mut file = create_dump_file_with_basename(tcx, &file_basename, "html")?;
            if body.source.def_id().is_local() {
                write_mir_fn_spanview(tcx, body, spanview, &file_basename, &mut file)?;
            }
        };
    }
}

/// Returns the file basename portion (without extension) of a filename path
/// where we should dump a MIR representation output files.
fn dump_file_basename<'tcx>(
    tcx: TyCtxt<'tcx>,
    pass_num: Option<&dyn Display>,
    pass_name: &str,
    disambiguator: &dyn Display,
    source: MirSource<'tcx>,
) -> String {
    let promotion_id = match source.promoted {
        Some(id) => format!("-{:?}", id),
        None => String::new(),
    };

    let pass_num = if tcx.sess.opts.debugging_opts.dump_mir_exclude_pass_number {
        String::new()
    } else {
        match pass_num {
            None => ".-------".to_string(),
            Some(pass_num) => format!(".{}", pass_num),
        }
    };

    let crate_name = tcx.crate_name(source.def_id().krate);
    let item_name = tcx.def_path(source.def_id()).to_filename_friendly_no_crate();
    // All drop shims have the same DefId, so we have to add the type
    // to get unique file names.
    let shim_disambiguator = match source.instance {
        ty::InstanceDef::DropGlue(_, Some(ty)) => {
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

    format!(
        "{}.{}{}{}{}.{}.{}",
        crate_name, item_name, shim_disambiguator, promotion_id, pass_num, pass_name, disambiguator,
    )
}

/// Returns the path to the filename where we should dump a given MIR.
/// Also used by other bits of code (e.g., NLL inference) that dump
/// graphviz data or other things.
fn dump_path(tcx: TyCtxt<'_>, basename: &str, extension: &str) -> PathBuf {
    let mut file_path = PathBuf::new();
    file_path.push(Path::new(&tcx.sess.opts.debugging_opts.dump_mir_dir));

    let file_name = format!("{}.{}", basename, extension,);

    file_path.push(&file_name);

    file_path
}

/// Attempts to open the MIR dump file with the given name and extension.
fn create_dump_file_with_basename(
    tcx: TyCtxt<'_>,
    file_basename: &str,
    extension: &str,
) -> io::Result<io::BufWriter<fs::File>> {
    let file_path = dump_path(tcx, file_basename, extension);
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("IO error creating MIR dump directory: {:?}; {}", parent, e),
            )
        })?;
    }
    Ok(io::BufWriter::new(fs::File::create(&file_path).map_err(|e| {
        io::Error::new(e.kind(), format!("IO error creating MIR dump file: {:?}; {}", file_path, e))
    })?))
}

/// Attempts to open a file where we should dump a given MIR or other
/// bit of MIR-related data. Used by `mir-dump`, but also by other
/// bits of code (e.g., NLL inference) that dump graphviz data or
/// other things, and hence takes the extension as an argument.
pub fn create_dump_file<'tcx>(
    tcx: TyCtxt<'tcx>,
    extension: &str,
    pass_num: Option<&dyn Display>,
    pass_name: &str,
    disambiguator: &dyn Display,
    source: MirSource<'tcx>,
) -> io::Result<io::BufWriter<fs::File>> {
    create_dump_file_with_basename(
        tcx,
        &dump_file_basename(tcx, pass_num, pass_name, disambiguator, source),
        extension,
    )
}

/// Write out a human-readable textual representation for the given MIR.
pub fn write_mir_pretty<'tcx>(
    tcx: TyCtxt<'tcx>,
    single: Option<DefId>,
    w: &mut dyn Write,
) -> io::Result<()> {
    writeln!(w, "// WARNING: This output format is intended for human consumers only")?;
    writeln!(w, "// and is subject to change without notice. Knock yourself out.")?;

    let mut first = true;
    for def_id in dump_mir_def_ids(tcx, single) {
        if first {
            first = false;
        } else {
            // Put empty lines between all items
            writeln!(w)?;
        }

        let render_body = |w: &mut dyn Write, body| -> io::Result<()> {
            write_mir_fn(tcx, body, &mut |_, _| Ok(()), w)?;

            for body in tcx.promoted_mir(def_id) {
                writeln!(w)?;
                write_mir_fn(tcx, body, &mut |_, _| Ok(()), w)?;
            }
            Ok(())
        };

        // For `const fn` we want to render both the optimized MIR and the MIR for ctfe.
        if tcx.is_const_fn_raw(def_id) {
            render_body(w, tcx.optimized_mir(def_id))?;
            writeln!(w)?;
            writeln!(w, "// MIR FOR CTFE")?;
            // Do not use `render_body`, as that would render the promoteds again, but these
            // are shared between mir_for_ctfe and optimized_mir
            write_mir_fn(tcx, tcx.mir_for_ctfe(def_id), &mut |_, _| Ok(()), w)?;
        } else {
            let instance_mir =
                tcx.instance_mir(ty::InstanceDef::Item(ty::WithOptConstParam::unknown(def_id)));
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
    w: &mut dyn Write,
) -> io::Result<()>
where
    F: FnMut(PassWhere, &mut dyn Write) -> io::Result<()>,
{
    write_mir_intro(tcx, body, w)?;
    for block in body.basic_blocks().indices() {
        extra_data(PassWhere::BeforeBlock(block), w)?;
        write_basic_block(tcx, block, body, extra_data, w)?;
        if block.index() + 1 != body.basic_blocks().len() {
            writeln!(w)?;
        }
    }

    writeln!(w, "}}")?;

    write_allocations(tcx, body, w)?;

    Ok(())
}

/// Write out a human-readable textual representation for the given basic block.
pub fn write_basic_block<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    block: BasicBlock,
    body: &Body<'tcx>,
    extra_data: &mut F,
    w: &mut dyn Write,
) -> io::Result<()>
where
    F: FnMut(PassWhere, &mut dyn Write) -> io::Result<()>,
{
    let data = &body[block];

    // Basic block label at the top.
    let cleanup_text = if data.is_cleanup { " (cleanup)" } else { "" };
    writeln!(w, "{}{:?}{}: {{", INDENT, block, cleanup_text)?;

    // List of statements in the middle.
    let mut current_location = Location { block, statement_index: 0 };
    for statement in &data.statements {
        extra_data(PassWhere::BeforeLocation(current_location), w)?;
        let indented_body = format!("{0}{0}{1:?};", INDENT, statement);
        writeln!(
            w,
            "{:A$} // {}{}",
            indented_body,
            if tcx.sess.verbose() { format!("{:?}: ", current_location) } else { String::new() },
            comment(tcx, statement.source_info),
            A = ALIGN,
        )?;

        write_extra(tcx, w, |visitor| {
            visitor.visit_statement(statement, current_location);
        })?;

        extra_data(PassWhere::AfterLocation(current_location), w)?;

        current_location.statement_index += 1;
    }

    // Terminator at the bottom.
    extra_data(PassWhere::BeforeLocation(current_location), w)?;
    let indented_terminator = format!("{0}{0}{1:?};", INDENT, data.terminator().kind);
    writeln!(
        w,
        "{:A$} // {}{}",
        indented_terminator,
        if tcx.sess.verbose() { format!("{:?}: ", current_location) } else { String::new() },
        comment(tcx, data.terminator().source_info),
        A = ALIGN,
    )?;

    write_extra(tcx, w, |visitor| {
        visitor.visit_terminator(data.terminator(), current_location);
    })?;

    extra_data(PassWhere::AfterLocation(current_location), w)?;
    extra_data(PassWhere::AfterTerminator(block), w)?;

    writeln!(w, "{}}}", INDENT)
}

/// After we print the main statement, we sometimes dump extra
/// information. There's often a lot of little things "nuzzled up" in
/// a statement.
fn write_extra<'tcx, F>(tcx: TyCtxt<'tcx>, write: &mut dyn Write, mut visit_op: F) -> io::Result<()>
where
    F: FnMut(&mut ExtraComments<'tcx>),
{
    let mut extra_comments = ExtraComments { tcx, comments: vec![] };
    visit_op(&mut extra_comments);
    for comment in extra_comments.comments {
        writeln!(write, "{:A$} // {}", "", comment, A = ALIGN)?;
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

fn use_verbose<'tcx>(ty: Ty<'tcx>, fn_def: bool) -> bool {
    match *ty.kind() {
        ty::Int(_) | ty::Uint(_) | ty::Bool | ty::Char | ty::Float(_) => false,
        // Unit type
        ty::Tuple(g_args) if g_args.is_empty() => false,
        ty::Tuple(g_args) => g_args.iter().any(|g_arg| use_verbose(g_arg.expect_ty(), fn_def)),
        ty::Array(ty, _) => use_verbose(ty, fn_def),
        ty::FnDef(..) => fn_def,
        _ => true,
    }
}

impl<'tcx> Visitor<'tcx> for ExtraComments<'tcx> {
    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        self.super_constant(constant, location);
        let Constant { span, user_ty, literal } = constant;
        if use_verbose(literal.ty(), true) {
            self.push("mir::Constant");
            self.push(&format!(
                "+ span: {}",
                self.tcx.sess.source_map().span_to_embeddable_string(*span)
            ));
            if let Some(user_ty) = user_ty {
                self.push(&format!("+ user_ty: {:?}", user_ty));
            }
            match literal {
                ConstantKind::Ty(literal) => self.push(&format!("+ literal: {:?}", literal)),
                ConstantKind::Val(val, ty) => {
                    // To keep the diffs small, we render this almost like we render ty::Const
                    self.push(&format!("+ literal: Const {{ ty: {}, val: Value({:?}) }}", ty, val))
                }
            }
        }
    }

    fn visit_const(&mut self, constant: ty::Const<'tcx>, _: Location) {
        self.super_const(constant);
        let ty = constant.ty();
        let val = constant.val();
        if use_verbose(ty, false) {
            self.push("ty::Const");
            self.push(&format!("+ ty: {:?}", ty));
            let val = match val {
                ty::ConstKind::Param(p) => format!("Param({})", p),
                ty::ConstKind::Infer(infer) => format!("Infer({:?})", infer),
                ty::ConstKind::Bound(idx, var) => format!("Bound({:?}, {:?})", idx, var),
                ty::ConstKind::Placeholder(ph) => format!("PlaceHolder({:?})", ph),
                ty::ConstKind::Unevaluated(uv) => format!(
                    "Unevaluated({}, {:?}, {:?})",
                    self.tcx.def_path_str(uv.def.did),
                    uv.substs,
                    uv.promoted,
                ),
                ty::ConstKind::Value(val) => format!("Value({:?})", val),
                ty::ConstKind::Error(_) => "Error".to_string(),
            };
            self.push(&format!("+ val: {}", val));
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        if let Rvalue::Aggregate(kind, _) = rvalue {
            match **kind {
                AggregateKind::Closure(def_id, substs) => {
                    self.push("closure");
                    self.push(&format!("+ def_id: {:?}", def_id));
                    self.push(&format!("+ substs: {:#?}", substs));
                }

                AggregateKind::Generator(def_id, substs, movability) => {
                    self.push("generator");
                    self.push(&format!("+ def_id: {:?}", def_id));
                    self.push(&format!("+ substs: {:#?}", substs));
                    self.push(&format!("+ movability: {:?}", movability));
                }

                AggregateKind::Adt(_, _, _, Some(user_ty), _) => {
                    self.push("adt");
                    self.push(&format!("+ user_ty: {:?}", user_ty));
                }

                _ => {}
            }
        }
    }
}

fn comment(tcx: TyCtxt<'_>, SourceInfo { span, scope }: SourceInfo) -> String {
    format!("scope {} at {}", scope.index(), tcx.sess.source_map().span_to_embeddable_string(span))
}

/// Prints local variables in a scope tree.
fn write_scope_tree(
    tcx: TyCtxt<'_>,
    body: &Body<'_>,
    scope_tree: &FxHashMap<SourceScope, Vec<SourceScope>>,
    w: &mut dyn Write,
    parent: SourceScope,
    depth: usize,
) -> io::Result<()> {
    let indent = depth * INDENT.len();

    // Local variable debuginfo.
    for var_debug_info in &body.var_debug_info {
        if var_debug_info.source_info.scope != parent {
            // Not declared in this scope.
            continue;
        }

        let indented_debug_info = format!(
            "{0:1$}debug {2} => {3:?};",
            INDENT, indent, var_debug_info.name, var_debug_info.value,
        );

        writeln!(
            w,
            "{0:1$} // in {2}",
            indented_debug_info,
            ALIGN,
            comment(tcx, var_debug_info.source_info),
        )?;
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

        let mut_str = if local_decl.mutability == Mutability::Mut { "mut " } else { "" };

        let mut indented_decl =
            format!("{0:1$}let {2}{3:?}: {4:?}", INDENT, indent, mut_str, local, local_decl.ty);
        if let Some(user_ty) = &local_decl.user_ty {
            for user_ty in user_ty.projections() {
                write!(indented_decl, " as {:?}", user_ty).unwrap();
            }
        }
        indented_decl.push(';');

        let local_name =
            if local == RETURN_PLACE { " return place".to_string() } else { String::new() };

        writeln!(
            w,
            "{0:1$} //{2} in {3}",
            indented_decl,
            ALIGN,
            local_name,
            comment(tcx, local_decl.source_info),
        )?;
    }

    let children = match scope_tree.get(&parent) {
        Some(children) => children,
        None => return Ok(()),
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

        if let Some(span) = span {
            writeln!(
                w,
                "{0:1$} // at {2}",
                indented_header,
                ALIGN,
                tcx.sess.source_map().span_to_embeddable_string(span),
            )?;
        } else {
            writeln!(w, "{}", indented_header)?;
        }

        write_scope_tree(tcx, body, scope_tree, w, child, depth + 1)?;
        writeln!(w, "{0:1$}}}", "", depth * INDENT.len())?;
    }

    Ok(())
}

/// Write out a human-readable textual representation of the MIR's `fn` type and the types of its
/// local variables (both user-defined bindings and compiler temporaries).
pub fn write_mir_intro<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'_>,
    w: &mut dyn Write,
) -> io::Result<()> {
    write_mir_sig(tcx, body, w)?;
    writeln!(w, "{{")?;

    // construct a scope tree and write it out
    let mut scope_tree: FxHashMap<SourceScope, Vec<SourceScope>> = Default::default();
    for (index, scope_data) in body.source_scopes.iter().enumerate() {
        if let Some(parent) = scope_data.parent_scope {
            scope_tree.entry(parent).or_default().push(SourceScope::new(index));
        } else {
            // Only the argument scope has no parent, because it's the root.
            assert_eq!(index, OUTERMOST_SOURCE_SCOPE.index());
        }
    }

    write_scope_tree(tcx, body, &scope_tree, w, OUTERMOST_SOURCE_SCOPE, 1)?;

    // Add an empty line before the first block is printed.
    writeln!(w)?;

    Ok(())
}

/// Find all `AllocId`s mentioned (recursively) in the MIR body and print their corresponding
/// allocations.
pub fn write_allocations<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'_>,
    w: &mut dyn Write,
) -> io::Result<()> {
    fn alloc_ids_from_alloc(alloc: &Allocation) -> impl DoubleEndedIterator<Item = AllocId> + '_ {
        alloc.relocations().values().map(|id| *id)
    }

    fn alloc_ids_from_const(val: ConstValue<'_>) -> impl Iterator<Item = AllocId> + '_ {
        match val {
            ConstValue::Scalar(interpret::Scalar::Ptr(ptr, _size)) => {
                Either::Left(Either::Left(std::iter::once(ptr.provenance)))
            }
            ConstValue::Scalar(interpret::Scalar::Int { .. }) => {
                Either::Left(Either::Right(std::iter::empty()))
            }
            ConstValue::ByRef { alloc, .. } | ConstValue::Slice { data: alloc, .. } => {
                Either::Right(alloc_ids_from_alloc(alloc))
            }
        }
    }

    struct CollectAllocIds(BTreeSet<AllocId>);

    impl<'tcx> Visitor<'tcx> for CollectAllocIds {
        fn visit_const(&mut self, c: ty::Const<'tcx>, _loc: Location) {
            if let ty::ConstKind::Value(val) = c.val() {
                self.0.extend(alloc_ids_from_const(val));
            }
        }

        fn visit_constant(&mut self, c: &Constant<'tcx>, loc: Location) {
            match c.literal {
                ConstantKind::Ty(c) => self.visit_const(c, loc),
                ConstantKind::Val(val, _) => {
                    self.0.extend(alloc_ids_from_const(val));
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
            |w: &mut dyn Write, alloc: &Allocation| -> io::Result<()> {
                // `.rev()` because we are popping them from the back of the `todo` vector.
                for id in alloc_ids_from_alloc(alloc).rev() {
                    if seen.insert(id) {
                        todo.push(id);
                    }
                }
                write!(w, "{}", display_allocation(tcx, alloc))
            };
        write!(w, "\n{}", id)?;
        match tcx.get_global_alloc(id) {
            // This can't really happen unless there are bugs, but it doesn't cost us anything to
            // gracefully handle it and allow buggy rustc to be debugged via allocation printing.
            None => write!(w, " (deallocated)")?,
            Some(GlobalAlloc::Function(inst)) => write!(w, " (fn: {})", inst)?,
            Some(GlobalAlloc::Static(did)) if !tcx.is_foreign_item(did) => {
                match tcx.eval_static_initializer(did) {
                    Ok(alloc) => {
                        write!(w, " (static: {}, ", tcx.def_path_str(did))?;
                        write_allocation_track_relocs(w, alloc)?;
                    }
                    Err(_) => write!(
                        w,
                        " (static: {}, error during initializer evaluation)",
                        tcx.def_path_str(did)
                    )?,
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
/// This also prints relocations adequately.
pub fn display_allocation<'a, 'tcx, Tag, Extra>(
    tcx: TyCtxt<'tcx>,
    alloc: &'a Allocation<Tag, Extra>,
) -> RenderAllocation<'a, 'tcx, Tag, Extra> {
    RenderAllocation { tcx, alloc }
}

#[doc(hidden)]
pub struct RenderAllocation<'a, 'tcx, Tag, Extra> {
    tcx: TyCtxt<'tcx>,
    alloc: &'a Allocation<Tag, Extra>,
}

impl<'a, 'tcx, Tag: Provenance, Extra> std::fmt::Display
    for RenderAllocation<'a, 'tcx, Tag, Extra>
{
    fn fmt(&self, w: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let RenderAllocation { tcx, alloc } = *self;
        write!(w, "size: {}, align: {})", alloc.size().bytes(), alloc.align.bytes())?;
        if alloc.size() == Size::ZERO {
            // We are done.
            return write!(w, " {{}}");
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
    writeln!(w, " │ {}", ascii)
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
fn write_allocation_bytes<'tcx, Tag: Provenance, Extra>(
    tcx: TyCtxt<'tcx>,
    alloc: &Allocation<Tag, Extra>,
    w: &mut dyn std::fmt::Write,
    prefix: &str,
) -> std::fmt::Result {
    let num_lines = alloc.size().bytes_usize().saturating_sub(BYTES_PER_LINE);
    // Number of chars needed to represent all line numbers.
    let pos_width = hex_number_length(alloc.size().bytes());

    if num_lines > 0 {
        write!(w, "{}0x{:02$x} │ ", prefix, 0, pos_width)?;
    } else {
        write!(w, "{}", prefix)?;
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
        if let Some(&tag) = alloc.relocations().get(&i) {
            // Memory with a relocation must be defined
            let j = i.bytes_usize();
            let offset = alloc
                .inspect_with_uninit_and_ptr_outside_interpreter(j..j + ptr_size.bytes_usize());
            let offset = read_target_uint(tcx.data_layout.endian, offset).unwrap();
            let offset = Size::from_bytes(offset);
            let relocation_width = |bytes| bytes * 3;
            let ptr = Pointer::new(tag, offset);
            let mut target = format!("{:?}", ptr);
            if target.len() > relocation_width(ptr_size.bytes_usize() - 1) {
                // This is too long, try to save some space.
                target = format!("{:#?}", ptr);
            }
            if ((i - line_start) + ptr_size).bytes_usize() > BYTES_PER_LINE {
                // This branch handles the situation where a relocation starts in the current line
                // but ends in the next one.
                let remainder = Size::from_bytes(BYTES_PER_LINE) - (i - line_start);
                let overflow = ptr_size - remainder;
                let remainder_width = relocation_width(remainder.bytes_usize()) - 2;
                let overflow_width = relocation_width(overflow.bytes_usize() - 1) + 1;
                ascii.push('╾');
                for _ in 0..remainder.bytes() - 1 {
                    ascii.push('─');
                }
                if overflow_width > remainder_width && overflow_width >= target.len() {
                    // The case where the relocation fits into the part in the next line
                    write!(w, "╾{0:─^1$}", "", remainder_width)?;
                    line_start =
                        write_allocation_newline(w, line_start, &ascii, pos_width, prefix)?;
                    ascii.clear();
                    write!(w, "{0:─^1$}╼", target, overflow_width)?;
                } else {
                    oversized_ptr(&mut target, remainder_width);
                    write!(w, "╾{0:─^1$}", target, remainder_width)?;
                    line_start =
                        write_allocation_newline(w, line_start, &ascii, pos_width, prefix)?;
                    write!(w, "{0:─^1$}╼", "", overflow_width)?;
                    ascii.clear();
                }
                for _ in 0..overflow.bytes() - 1 {
                    ascii.push('─');
                }
                ascii.push('╼');
                i += ptr_size;
                continue;
            } else {
                // This branch handles a relocation that starts and ends in the current line.
                let relocation_width = relocation_width(ptr_size.bytes_usize() - 1);
                oversized_ptr(&mut target, relocation_width);
                ascii.push('╾');
                write!(w, "╾{0:─^1$}╼", target, relocation_width)?;
                for _ in 0..ptr_size.bytes() - 2 {
                    ascii.push('─');
                }
                ascii.push('╼');
                i += ptr_size;
            }
        } else if alloc.init_mask().is_range_initialized(i, i + Size::from_bytes(1)).is_ok() {
            let j = i.bytes_usize();

            // Checked definedness (and thus range) and relocations. This access also doesn't
            // influence interpreter execution but is only for debugging.
            let c = alloc.inspect_with_uninit_and_ptr_outside_interpreter(j..j + 1)[0];
            write!(w, "{:02x}", c)?;
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

fn write_mir_sig(tcx: TyCtxt<'_>, body: &Body<'_>, w: &mut dyn Write) -> io::Result<()> {
    use rustc_hir::def::DefKind;

    trace!("write_mir_sig: {:?}", body.source.instance);
    let def_id = body.source.def_id();
    let kind = tcx.def_kind(def_id);
    let is_function = match kind {
        DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(..) => true,
        _ => tcx.is_closure(def_id),
    };
    match (kind, body.source.promoted) {
        (_, Some(i)) => write!(w, "{:?} in ", i)?,
        (DefKind::Const | DefKind::AssocConst, _) => write!(w, "const ")?,
        (DefKind::Static, _) => {
            write!(w, "static {}", if tcx.is_mutable_static(def_id) { "mut " } else { "" })?
        }
        (_, _) if is_function => write!(w, "fn ")?,
        (DefKind::AnonConst | DefKind::InlineConst, _) => {} // things like anon const, not an item
        _ => bug!("Unexpected def kind {:?}", kind),
    }

    ty::print::with_forced_impl_filename_line(|| {
        // see notes on #41697 elsewhere
        write!(w, "{}", tcx.def_path_str(def_id))
    })?;

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
        writeln!(w, "yields {}", yield_ty)?;
    }

    write!(w, " ")?;
    // Next thing that gets printed is the opening {

    Ok(())
}

fn write_user_type_annotations(
    tcx: TyCtxt<'_>,
    body: &Body<'_>,
    w: &mut dyn Write,
) -> io::Result<()> {
    if !body.user_type_annotations.is_empty() {
        writeln!(w, "| User Type Annotations")?;
    }
    for (index, annotation) in body.user_type_annotations.iter_enumerated() {
        writeln!(
            w,
            "| {:?}: {:?} at {}",
            index.index(),
            annotation.user_ty,
            tcx.sess.source_map().span_to_embeddable_string(annotation.span)
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

/// Calc converted u64 decimal into hex and return it's length in chars
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
