//! Dataflow analysis results.

use std::ffi::OsString;
use std::path::PathBuf;

use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::mir::{self, BasicBlock, create_dump_file, dump_enabled, traversal};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_span::symbol::{Symbol, sym};
use tracing::debug;
use {rustc_ast as ast, rustc_graphviz as dot};

use super::fmt::DebugWithContext;
use super::{Analysis, ResultsCursor, ResultsVisitor, graphviz, visit_results};
use crate::errors::{
    DuplicateValuesFor, PathMustEndInFilename, RequiresAnArgument, UnknownFormatter,
};
use crate::framework::cursor::ResultsHandle;

pub type EntrySets<'tcx, A> = IndexVec<BasicBlock, <A as Analysis<'tcx>>::Domain>;

/// A dataflow analysis that has converged to fixpoint. It only holds the domain values at the
/// entry of each basic block. Domain values in other parts of the block are recomputed on the fly
/// by visitors (i.e. `ResultsCursor`, or `ResultsVisitor` impls).
#[derive(Clone)]
pub struct Results<'tcx, A>
where
    A: Analysis<'tcx>,
{
    pub analysis: A,
    pub entry_sets: EntrySets<'tcx, A>,
}

impl<'tcx, A> Results<'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// Creates a `ResultsCursor` that mutably borrows the `Results`, which is appropriate when the
    /// `Results` is also used outside the cursor.
    pub fn as_results_cursor<'mir>(
        &'mir mut self,
        body: &'mir mir::Body<'tcx>,
    ) -> ResultsCursor<'mir, 'tcx, A> {
        ResultsCursor::new(body, ResultsHandle::BorrowedMut(self))
    }

    /// Creates a `ResultsCursor` that takes ownership of the `Results`.
    pub fn into_results_cursor<'mir>(
        self,
        body: &'mir mir::Body<'tcx>,
    ) -> ResultsCursor<'mir, 'tcx, A> {
        ResultsCursor::new(body, ResultsHandle::Owned(self))
    }

    /// Gets the dataflow state for the given block.
    pub fn entry_set_for_block(&self, block: BasicBlock) -> &A::Domain {
        &self.entry_sets[block]
    }

    pub fn visit_with<'mir>(
        &mut self,
        body: &'mir mir::Body<'tcx>,
        blocks: impl IntoIterator<Item = BasicBlock>,
        vis: &mut impl ResultsVisitor<'mir, 'tcx, A>,
    ) {
        visit_results(body, blocks, self, vis)
    }

    pub fn visit_reachable_with<'mir>(
        &mut self,
        body: &'mir mir::Body<'tcx>,
        vis: &mut impl ResultsVisitor<'mir, 'tcx, A>,
    ) {
        let blocks = traversal::reachable(body);
        visit_results(body, blocks.map(|(bb, _)| bb), self, vis)
    }
}

// Graphviz

/// Writes a DOT file containing the results of a dataflow analysis if the user requested it via
/// `rustc_mir` attributes and `-Z dump-mir-dataflow`. The `Result` in and the `Results` out are
/// the same.
pub(super) fn write_graphviz_results<'tcx, A>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    results: &mut Results<'tcx, A>,
    pass_name: Option<&'static str>,
) -> std::io::Result<()>
where
    A: Analysis<'tcx>,
    A::Domain: DebugWithContext<A>,
{
    use std::fs;
    use std::io::Write;

    let def_id = body.source.def_id();
    let Ok(attrs) = RustcMirAttrs::parse(tcx, def_id) else {
        // Invalid `rustc_mir` attrs are reported in `RustcMirAttrs::parse`
        return Ok(());
    };

    let file = try {
        match attrs.output_path(A::NAME) {
            Some(path) => {
                debug!("printing dataflow results for {:?} to {}", def_id, path.display());
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::File::create_buffered(&path)?
            }

            None if dump_enabled(tcx, A::NAME, def_id) => {
                create_dump_file(tcx, "dot", false, A::NAME, &pass_name.unwrap_or("-----"), body)?
            }

            _ => return Ok(()),
        }
    };
    let mut file = match file {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    let style = match attrs.formatter {
        Some(sym::two_phase) => graphviz::OutputStyle::BeforeAndAfter,
        _ => graphviz::OutputStyle::AfterOnly,
    };

    let mut buf = Vec::new();

    let graphviz = graphviz::Formatter::new(body, results, style);
    let mut render_opts =
        vec![dot::RenderOption::Fontname(tcx.sess.opts.unstable_opts.graphviz_font.clone())];
    if tcx.sess.opts.unstable_opts.graphviz_dark_mode {
        render_opts.push(dot::RenderOption::DarkTheme);
    }
    let r = with_no_trimmed_paths!(dot::render_opts(&graphviz, &mut buf, &render_opts));

    let lhs = try {
        r?;
        file.write_all(&buf)?;
    };

    lhs
}

#[derive(Default)]
struct RustcMirAttrs {
    basename_and_suffix: Option<PathBuf>,
    formatter: Option<Symbol>,
}

impl RustcMirAttrs {
    fn parse(tcx: TyCtxt<'_>, def_id: DefId) -> Result<Self, ()> {
        let mut result = Ok(());
        let mut ret = RustcMirAttrs::default();

        let rustc_mir_attrs = tcx
            .get_attrs(def_id, sym::rustc_mir)
            .flat_map(|attr| attr.meta_item_list().into_iter().flat_map(|v| v.into_iter()));

        for attr in rustc_mir_attrs {
            let attr_result = if attr.has_name(sym::borrowck_graphviz_postflow) {
                Self::set_field(&mut ret.basename_and_suffix, tcx, &attr, |s| {
                    let path = PathBuf::from(s.to_string());
                    match path.file_name() {
                        Some(_) => Ok(path),
                        None => {
                            tcx.dcx().emit_err(PathMustEndInFilename { span: attr.span() });
                            Err(())
                        }
                    }
                })
            } else if attr.has_name(sym::borrowck_graphviz_format) {
                Self::set_field(&mut ret.formatter, tcx, &attr, |s| match s {
                    sym::gen_kill | sym::two_phase => Ok(s),
                    _ => {
                        tcx.dcx().emit_err(UnknownFormatter { span: attr.span() });
                        Err(())
                    }
                })
            } else {
                Ok(())
            };

            result = result.and(attr_result);
        }

        result.map(|()| ret)
    }

    fn set_field<T>(
        field: &mut Option<T>,
        tcx: TyCtxt<'_>,
        attr: &ast::MetaItemInner,
        mapper: impl FnOnce(Symbol) -> Result<T, ()>,
    ) -> Result<(), ()> {
        if field.is_some() {
            tcx.dcx()
                .emit_err(DuplicateValuesFor { span: attr.span(), name: attr.name_or_empty() });

            return Err(());
        }

        if let Some(s) = attr.value_str() {
            *field = Some(mapper(s)?);
            Ok(())
        } else {
            tcx.dcx()
                .emit_err(RequiresAnArgument { span: attr.span(), name: attr.name_or_empty() });
            Err(())
        }
    }

    /// Returns the path where dataflow results should be written, or `None`
    /// `borrowck_graphviz_postflow` was not specified.
    ///
    /// This performs the following transformation to the argument of `borrowck_graphviz_postflow`:
    ///
    /// "path/suffix.dot" -> "path/analysis_name_suffix.dot"
    fn output_path(&self, analysis_name: &str) -> Option<PathBuf> {
        let mut ret = self.basename_and_suffix.as_ref().cloned()?;
        let suffix = ret.file_name().unwrap(); // Checked when parsing attrs

        let mut file_name: OsString = analysis_name.into();
        file_name.push("_");
        file_name.push(suffix);
        ret.set_file_name(file_name);

        Some(ret)
    }
}
