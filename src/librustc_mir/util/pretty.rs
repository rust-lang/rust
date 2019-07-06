use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::mir::*;
use rustc::mir::visit::Visitor;
use rustc::ty::{self, TyCtxt};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use std::fmt::Display;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use super::graphviz::write_mir_fn_graphviz;
use crate::transform::MirSource;

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
pub fn dump_mir<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    pass_num: Option<&dyn Display>,
    pass_name: &str,
    disambiguator: &dyn Display,
    source: MirSource<'tcx>,
    body: &Body<'tcx>,
    extra_data: F,
) where
    F: FnMut(PassWhere, &mut dyn Write) -> io::Result<()>,
{
    if !dump_enabled(tcx, pass_name, source) {
        return;
    }

    let node_path = ty::print::with_forced_impl_filename_line(|| {
        // see notes on #41697 below
        tcx.def_path_str(source.def_id())
    });
    dump_matched_mir_node(
        tcx,
        pass_num,
        pass_name,
        &node_path,
        disambiguator,
        source,
        body,
        extra_data,
    );
}

pub fn dump_enabled<'tcx>(tcx: TyCtxt<'tcx>, pass_name: &str, source: MirSource<'tcx>) -> bool {
    let filters = match tcx.sess.opts.debugging_opts.dump_mir {
        None => return false,
        Some(ref filters) => filters,
    };
    let node_path = ty::print::with_forced_impl_filename_line(|| {
        // see notes on #41697 below
        tcx.def_path_str(source.def_id())
    });
    filters.split('|').any(|or_filter| {
        or_filter.split('&').all(|and_filter| {
            and_filter == "all" || pass_name.contains(and_filter) || node_path.contains(and_filter)
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
    node_path: &str,
    disambiguator: &dyn Display,
    source: MirSource<'tcx>,
    body: &Body<'tcx>,
    mut extra_data: F,
) where
    F: FnMut(PassWhere, &mut dyn Write) -> io::Result<()>,
{
    let _: io::Result<()> = try {
        let mut file = create_dump_file(tcx, "mir", pass_num, pass_name, disambiguator, source)?;
        writeln!(file, "// MIR for `{}`", node_path)?;
        writeln!(file, "// source = {:?}", source)?;
        writeln!(file, "// pass_name = {}", pass_name)?;
        writeln!(file, "// disambiguator = {}", disambiguator)?;
        if let Some(ref layout) = body.generator_layout {
            writeln!(file, "// generator_layout = {:?}", layout)?;
        }
        writeln!(file, "")?;
        extra_data(PassWhere::BeforeCFG, &mut file)?;
        write_user_type_annotations(body, &mut file)?;
        write_mir_fn(tcx, source, body, &mut extra_data, &mut file)?;
        extra_data(PassWhere::AfterCFG, &mut file)?;
    };

    if tcx.sess.opts.debugging_opts.dump_mir_graphviz {
        let _: io::Result<()> = try {
            let mut file =
                create_dump_file(tcx, "dot", pass_num, pass_name, disambiguator, source)?;
            write_mir_fn_graphviz(tcx, source.def_id(), body, &mut file)?;
        };
    }
}

/// Returns the path to the filename where we should dump a given MIR.
/// Also used by other bits of code (e.g., NLL inference) that dump
/// graphviz data or other things.
fn dump_path(
    tcx: TyCtxt<'_>,
    extension: &str,
    pass_num: Option<&dyn Display>,
    pass_name: &str,
    disambiguator: &dyn Display,
    source: MirSource<'tcx>,
) -> PathBuf {
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

    let mut file_path = PathBuf::new();
    file_path.push(Path::new(&tcx.sess.opts.debugging_opts.dump_mir_dir));

    let item_name = tcx
        .def_path(source.def_id())
        .to_filename_friendly_no_crate();
    // All drop shims have the same DefId, so we have to add the type
    // to get unique file names.
    let shim_disambiguator = match source.instance {
        ty::InstanceDef::DropGlue(_, Some(ty)) => {
            // Unfortunately, pretty-printed typed are not very filename-friendly.
            // We dome some filtering.
            let mut s = ".".to_owned();
            s.extend(ty.to_string()
                .chars()
                .filter_map(|c| match c {
                    ' ' => None,
                    ':' | '<' | '>' => Some('_'),
                    c => Some(c)
                }));
            s
        }
        _ => String::new(),
    };

    let file_name = format!(
        "rustc.{}{}{}{}.{}.{}.{}",
        item_name,
        shim_disambiguator,
        promotion_id,
        pass_num,
        pass_name,
        disambiguator,
        extension,
    );

    file_path.push(&file_name);

    file_path
}

/// Attempts to open a file where we should dump a given MIR or other
/// bit of MIR-related data. Used by `mir-dump`, but also by other
/// bits of code (e.g., NLL inference) that dump graphviz data or
/// other things, and hence takes the extension as an argument.
pub(crate) fn create_dump_file(
    tcx: TyCtxt<'_>,
    extension: &str,
    pass_num: Option<&dyn Display>,
    pass_name: &str,
    disambiguator: &dyn Display,
    source: MirSource<'tcx>,
) -> io::Result<fs::File> {
    let file_path = dump_path(tcx, extension, pass_num, pass_name, disambiguator, source);
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::File::create(&file_path)
}

/// Write out a human-readable textual representation for the given MIR.
pub fn write_mir_pretty<'tcx>(
    tcx: TyCtxt<'tcx>,
    single: Option<DefId>,
    w: &mut dyn Write,
) -> io::Result<()> {
    writeln!(
        w,
        "// WARNING: This output format is intended for human consumers only"
    )?;
    writeln!(
        w,
        "// and is subject to change without notice. Knock yourself out."
    )?;

    let mut first = true;
    for def_id in dump_mir_def_ids(tcx, single) {
        let body = &tcx.optimized_mir(def_id);

        if first {
            first = false;
        } else {
            // Put empty lines between all items
            writeln!(w, "")?;
        }

        write_mir_fn(tcx, MirSource::item(def_id), body, &mut |_, _| Ok(()), w)?;

        for (i, body) in body.promoted.iter_enumerated() {
            writeln!(w, "")?;
            let src = MirSource {
                instance: ty::InstanceDef::Item(def_id),
                promoted: Some(i),
            };
            write_mir_fn(tcx, src, body, &mut |_, _| Ok(()), w)?;
        }
    }
    Ok(())
}

pub fn write_mir_fn<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    src: MirSource<'tcx>,
    body: &Body<'tcx>,
    extra_data: &mut F,
    w: &mut dyn Write,
) -> io::Result<()>
where
    F: FnMut(PassWhere, &mut dyn Write) -> io::Result<()>,
{
    write_mir_intro(tcx, src, body, w)?;
    for block in body.basic_blocks().indices() {
        extra_data(PassWhere::BeforeBlock(block), w)?;
        write_basic_block(tcx, block, body, extra_data, w)?;
        if block.index() + 1 != body.basic_blocks().len() {
            writeln!(w, "")?;
        }
    }

    writeln!(w, "}}")?;
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
    let mut current_location = Location {
        block: block,
        statement_index: 0,
    };
    for statement in &data.statements {
        extra_data(PassWhere::BeforeLocation(current_location), w)?;
        let indented_body = format!("{0}{0}{1:?};", INDENT, statement);
        writeln!(
            w,
            "{:A$} // {:?}: {}",
            indented_body,
            current_location,
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
        "{:A$} // {:?}: {}",
        indented_terminator,
        current_location,
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
    let mut extra_comments = ExtraComments {
        _tcx: tcx,
        comments: vec![],
    };
    visit_op(&mut extra_comments);
    for comment in extra_comments.comments {
        writeln!(write, "{:A$} // {}", "", comment, A = ALIGN)?;
    }
    Ok(())
}

struct ExtraComments<'tcx> {
    _tcx: TyCtxt<'tcx>, // don't need it now, but bet we will soon
    comments: Vec<String>,
}

impl ExtraComments<'tcx> {
    fn push(&mut self, lines: &str) {
        for line in lines.split('\n') {
            self.comments.push(line.to_string());
        }
    }
}

impl Visitor<'tcx> for ExtraComments<'tcx> {
    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        self.super_constant(constant, location);
        let Constant { span, ty, user_ty, literal } = constant;
        self.push("mir::Constant");
        self.push(&format!("+ span: {:?}", span));
        self.push(&format!("+ ty: {:?}", ty));
        if let Some(user_ty) = user_ty {
            self.push(&format!("+ user_ty: {:?}", user_ty));
        }
        self.push(&format!("+ literal: {:?}", literal));
    }

    fn visit_const(&mut self, constant: &&'tcx ty::Const<'tcx>, _: Location) {
        self.super_const(constant);
        let ty::Const { ty, val, .. } = constant;
        self.push("ty::Const");
        self.push(&format!("+ ty: {:?}", ty));
        self.push(&format!("+ val: {:?}", val));
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        match rvalue {
            Rvalue::Aggregate(kind, _) => match **kind {
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
            },

            _ => {}
        }
    }
}

fn comment(tcx: TyCtxt<'_>, SourceInfo { span, scope }: SourceInfo) -> String {
    format!(
        "scope {} at {}",
        scope.index(),
        tcx.sess.source_map().span_to_string(span)
    )
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

    // Local variable types (including the user's name in a comment).
    for (local, local_decl) in body.local_decls.iter_enumerated() {
        if (1..body.arg_count+1).contains(&local.index()) {
            // Skip over argument locals, they're printed in the signature.
            continue;
        }

        if local_decl.source_info.scope != parent {
            // Not declared in this scope.
            continue;
        }

        let mut_str = if local_decl.mutability == Mutability::Mut {
            "mut "
        } else {
            ""
        };

        let mut indented_decl = format!(
            "{0:1$}let {2}{3:?}: {4:?}",
            INDENT,
            indent,
            mut_str,
            local,
            local_decl.ty
        );
        for user_ty in local_decl.user_ty.projections() {
            write!(indented_decl, " as {:?}", user_ty).unwrap();
        }
        indented_decl.push_str(";");

        let local_name = if local == RETURN_PLACE {
            format!(" return place")
        } else if let Some(name) = local_decl.name {
            format!(" \"{}\"", name)
        } else {
            String::new()
        };

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
        Some(childs) => childs,
        None => return Ok(()),
    };

    for &child in children {
        assert_eq!(body.source_scopes[child].parent_scope, Some(parent));
        writeln!(w, "{0:1$}scope {2} {{", "", indent, child.index())?;
        write_scope_tree(tcx, body, scope_tree, w, child, depth + 1)?;
        writeln!(w, "{0:1$}}}", "", depth * INDENT.len())?;
    }

    Ok(())
}

/// Write out a human-readable textual representation of the MIR's `fn` type and the types of its
/// local variables (both user-defined bindings and compiler temporaries).
pub fn write_mir_intro<'tcx>(
    tcx: TyCtxt<'tcx>,
    src: MirSource<'tcx>,
    body: &Body<'_>,
    w: &mut dyn Write,
) -> io::Result<()> {
    write_mir_sig(tcx, src, body, w)?;
    writeln!(w, "{{")?;

    // construct a scope tree and write it out
    let mut scope_tree: FxHashMap<SourceScope, Vec<SourceScope>> = Default::default();
    for (index, scope_data) in body.source_scopes.iter().enumerate() {
        if let Some(parent) = scope_data.parent_scope {
            scope_tree
                .entry(parent)
                .or_default()
                .push(SourceScope::new(index));
        } else {
            // Only the argument scope has no parent, because it's the root.
            assert_eq!(index, OUTERMOST_SOURCE_SCOPE.index());
        }
    }

    write_scope_tree(tcx, body, &scope_tree, w, OUTERMOST_SOURCE_SCOPE, 1)?;

    // Add an empty line before the first block is printed.
    writeln!(w, "")?;

    Ok(())
}

fn write_mir_sig(
    tcx: TyCtxt<'_>,
    src: MirSource<'tcx>,
    body: &Body<'_>,
    w: &mut dyn Write,
) -> io::Result<()> {
    use rustc::hir::def::DefKind;

    trace!("write_mir_sig: {:?}", src.instance);
    let kind = tcx.def_kind(src.def_id());
    let is_function = match kind {
        Some(DefKind::Fn)
        | Some(DefKind::Method)
        | Some(DefKind::Ctor(..)) => true,
        _ => tcx.is_closure(src.def_id()),
    };
    match (kind, src.promoted) {
        (_, Some(i)) => write!(w, "{:?} in ", i)?,
        (Some(DefKind::Const), _)
        | (Some(DefKind::AssocConst), _) => write!(w, "const ")?,
        (Some(DefKind::Static), _) =>
            write!(w, "static {}", if tcx.is_mutable_static(src.def_id()) { "mut " } else { "" })?,
        (_, _) if is_function => write!(w, "fn ")?,
        (None, _) => {}, // things like anon const, not an item
        _ => bug!("Unexpected def kind {:?}", kind),
    }

    ty::print::with_forced_impl_filename_line(|| {
        // see notes on #41697 elsewhere
        write!(w, " {}", tcx.def_path_str(src.def_id()))
    })?;

    if src.promoted.is_none() && is_function {
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

    if let Some(yield_ty) = body.yield_ty {
        writeln!(w)?;
        writeln!(w, "yields {}", yield_ty)?;
    }

    write!(w, " ")?;
    // Next thing that gets printed is the opening {

    Ok(())
}

fn write_user_type_annotations(body: &Body<'_>, w: &mut dyn Write) -> io::Result<()> {
    if !body.user_type_annotations.is_empty() {
        writeln!(w, "| User Type Annotations")?;
    }
    for (index, annotation) in body.user_type_annotations.iter_enumerated() {
        writeln!(w, "| {:?}: {:?} at {:?}", index.index(), annotation.user_ty, annotation.span)?;
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
        tcx.mir_keys(LOCAL_CRATE).iter().cloned().collect()
    }
}
