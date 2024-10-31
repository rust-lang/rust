use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::sync::Arc;

use rustc_ast as ast;
use rustc_codegen_ssa::CodegenResults;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::steal::Steal;
use rustc_data_structures::svh::Svh;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::dep_graph::DepGraph;
use rustc_middle::ty::{GlobalCtxt, TyCtxt};
use rustc_session::Session;
use rustc_session::config::{self, OutputFilenames, OutputType};

use crate::errors::FailedWritingFile;
use crate::interface::Compiler;
use crate::passes;

/// Represent the result of a query.
///
/// This result can be stolen once with the [`steal`] method and generated with the [`compute`] method.
///
/// [`steal`]: Steal::steal
/// [`compute`]: Self::compute
pub struct Query<T> {
    /// `None` means no value has been computed yet.
    result: RefCell<Option<Steal<T>>>,
}

impl<T> Query<T> {
    fn compute<F: FnOnce() -> T>(&self, f: F) -> QueryResult<'_, T> {
        QueryResult(RefMut::map(
            self.result.borrow_mut(),
            |r: &mut Option<Steal<T>>| -> &mut Steal<T> {
                r.get_or_insert_with(|| Steal::new(f()))
            },
        ))
    }
}

pub struct QueryResult<'a, T>(RefMut<'a, Steal<T>>);

impl<'a, T> std::ops::Deref for QueryResult<'a, T> {
    type Target = RefMut<'a, Steal<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> std::ops::DerefMut for QueryResult<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, 'tcx> QueryResult<'a, &'tcx GlobalCtxt<'tcx>> {
    pub fn enter<T>(&mut self, f: impl FnOnce(TyCtxt<'tcx>) -> T) -> T {
        (*self.0).borrow().enter(f)
    }
}

pub struct Queries<'tcx> {
    compiler: &'tcx Compiler,

    parse: Query<ast::Crate>,
}

impl<'tcx> Queries<'tcx> {
    pub fn new(compiler: &'tcx Compiler) -> Queries<'tcx> {
        Queries { compiler, parse: Query { result: RefCell::new(None) } }
    }

    pub fn parse(&self) -> QueryResult<'_, ast::Crate> {
        self.parse.compute(|| passes::parse(&self.compiler.sess))
    }
}

pub struct Linker {
    dep_graph: DepGraph,
    output_filenames: Arc<OutputFilenames>,
    // Only present when incr. comp. is enabled.
    crate_hash: Option<Svh>,
    ongoing_codegen: Box<dyn Any>,
}

impl Linker {
    pub fn codegen_and_build_linker(
        tcx: TyCtxt<'_>,
        codegen_backend: &dyn CodegenBackend,
    ) -> Linker {
        let ongoing_codegen = passes::start_codegen(codegen_backend, tcx);

        Linker {
            dep_graph: tcx.dep_graph.clone(),
            output_filenames: Arc::clone(tcx.output_filenames(())),
            crate_hash: if tcx.needs_crate_hash() {
                Some(tcx.crate_hash(LOCAL_CRATE))
            } else {
                None
            },
            ongoing_codegen,
        }
    }

    pub fn link(self, sess: &Session, codegen_backend: &dyn CodegenBackend) {
        let (codegen_results, work_products) = sess.time("finish_ongoing_codegen", || {
            codegen_backend.join_codegen(self.ongoing_codegen, sess, &self.output_filenames)
        });

        sess.dcx().abort_if_errors();

        let _timer = sess.timer("link");

        sess.time("serialize_work_products", || {
            rustc_incremental::save_work_product_index(sess, &self.dep_graph, work_products)
        });

        let prof = sess.prof.clone();
        prof.generic_activity("drop_dep_graph").run(move || drop(self.dep_graph));

        // Now that we won't touch anything in the incremental compilation directory
        // any more, we can finalize it (which involves renaming it)
        rustc_incremental::finalize_session_directory(sess, self.crate_hash);

        if !sess
            .opts
            .output_types
            .keys()
            .any(|&i| i == OutputType::Exe || i == OutputType::Metadata)
        {
            return;
        }

        if sess.opts.unstable_opts.no_link {
            let rlink_file = self.output_filenames.with_extension(config::RLINK_EXT);
            CodegenResults::serialize_rlink(
                sess,
                &rlink_file,
                &codegen_results,
                &*self.output_filenames,
            )
            .unwrap_or_else(|error| {
                sess.dcx().emit_fatal(FailedWritingFile { path: &rlink_file, error })
            });
            return;
        }

        let _timer = sess.prof.verbose_generic_activity("link_crate");
        codegen_backend.link(sess, codegen_results, &self.output_filenames)
    }
}

impl Compiler {
    pub fn enter<F, T>(&self, f: F) -> T
    where
        F: for<'tcx> FnOnce(&'tcx Queries<'tcx>) -> T,
    {
        let queries = Queries::new(self);
        f(&queries)
    }
}
