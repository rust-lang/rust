use crate::errors::{FailedWritingFile, RustcErrorFatal, RustcErrorUnexpectedAnnotation};
use crate::interface::{Compiler, Result};
use crate::passes::{self, BoxedResolver};

use rustc_ast as ast;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::CodegenResults;
use rustc_data_structures::steal::Steal;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{Lrc, OnceCell, WorkerLocal};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_incremental::DepGraphFuture;
use rustc_lint::LintStore;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::DepGraph;
use rustc_middle::ty::{self, GlobalCtxt, TyCtxt};
use rustc_query_impl::Queries as TcxQueries;
use rustc_session::config::{self, OutputFilenames, OutputType};
use rustc_session::{output::find_crate_name, Session};
use rustc_span::symbol::sym;
use rustc_span::Symbol;
use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::rc::Rc;
use std::sync::Arc;

/// Represent the result of a query.
///
/// This result can be stolen once with the [`steal`] method and generated with the [`compute`] method.
///
/// [`steal`]: Steal::steal
/// [`compute`]: Self::compute
pub struct Query<T> {
    /// `None` means no value has been computed yet.
    result: RefCell<Option<Result<Steal<T>>>>,
}

impl<T> Query<T> {
    fn compute<F: FnOnce() -> Result<T>>(&self, f: F) -> Result<QueryResult<'_, T>> {
        RefMut::filter_map(
            self.result.borrow_mut(),
            |r: &mut Option<Result<Steal<T>>>| -> Option<&mut Steal<T>> {
                r.get_or_insert_with(|| f().map(Steal::new)).as_mut().ok()
            },
        )
        .map_err(|r| *r.as_ref().unwrap().as_ref().map(|_| ()).unwrap_err())
        .map(QueryResult)
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
        (*self.0).get_mut().enter(f)
    }
}

impl<T> Default for Query<T> {
    fn default() -> Self {
        Query { result: RefCell::new(None) }
    }
}

pub struct Queries<'tcx> {
    compiler: &'tcx Compiler,
    gcx_cell: OnceCell<GlobalCtxt<'tcx>>,
    queries: OnceCell<TcxQueries<'tcx>>,

    arena: WorkerLocal<Arena<'tcx>>,
    hir_arena: WorkerLocal<rustc_hir::Arena<'tcx>>,

    dep_graph_future: Query<Option<DepGraphFuture>>,
    parse: Query<ast::Crate>,
    crate_name: Query<Symbol>,
    register_plugins: Query<(ast::Crate, Lrc<LintStore>)>,
    expansion: Query<(Lrc<ast::Crate>, Rc<RefCell<BoxedResolver>>, Lrc<LintStore>)>,
    dep_graph: Query<DepGraph>,
    // This just points to what's in `gcx_cell`.
    gcx: Query<&'tcx GlobalCtxt<'tcx>>,
    ongoing_codegen: Query<Box<dyn Any>>,
}

impl<'tcx> Queries<'tcx> {
    pub fn new(compiler: &'tcx Compiler) -> Queries<'tcx> {
        Queries {
            compiler,
            gcx_cell: OnceCell::new(),
            queries: OnceCell::new(),
            arena: WorkerLocal::new(|_| Arena::default()),
            hir_arena: WorkerLocal::new(|_| rustc_hir::Arena::default()),
            dep_graph_future: Default::default(),
            parse: Default::default(),
            crate_name: Default::default(),
            register_plugins: Default::default(),
            expansion: Default::default(),
            dep_graph: Default::default(),
            gcx: Default::default(),
            ongoing_codegen: Default::default(),
        }
    }

    fn session(&self) -> &Lrc<Session> {
        &self.compiler.sess
    }
    fn codegen_backend(&self) -> &Lrc<Box<dyn CodegenBackend>> {
        self.compiler.codegen_backend()
    }

    fn dep_graph_future(&self) -> Result<QueryResult<'_, Option<DepGraphFuture>>> {
        self.dep_graph_future.compute(|| {
            let sess = self.session();
            Ok(sess.opts.build_dep_graph().then(|| rustc_incremental::load_dep_graph(sess)))
        })
    }

    pub fn parse(&self) -> Result<QueryResult<'_, ast::Crate>> {
        self.parse
            .compute(|| passes::parse(self.session()).map_err(|mut parse_error| parse_error.emit()))
    }

    pub fn register_plugins(&self) -> Result<QueryResult<'_, (ast::Crate, Lrc<LintStore>)>> {
        self.register_plugins.compute(|| {
            let crate_name = *self.crate_name()?.borrow();
            let krate = self.parse()?.steal();

            let empty: &(dyn Fn(&Session, &mut LintStore) + Sync + Send) = &|_, _| {};
            let (krate, lint_store) = passes::register_plugins(
                self.session(),
                &*self.codegen_backend().metadata_loader(),
                self.compiler.register_lints.as_deref().unwrap_or_else(|| empty),
                krate,
                crate_name,
            )?;

            // Compute the dependency graph (in the background). We want to do
            // this as early as possible, to give the DepGraph maximum time to
            // load before dep_graph() is called, but it also can't happen
            // until after rustc_incremental::prepare_session_directory() is
            // called, which happens within passes::register_plugins().
            self.dep_graph_future().ok();

            Ok((krate, Lrc::new(lint_store)))
        })
    }

    fn crate_name(&self) -> Result<QueryResult<'_, Symbol>> {
        self.crate_name.compute(|| {
            Ok({
                let parse_result = self.parse()?;
                let krate = parse_result.borrow();
                // parse `#[crate_name]` even if `--crate-name` was passed, to make sure it matches.
                find_crate_name(self.session(), &krate.attrs)
            })
        })
    }

    pub fn expansion(
        &self,
    ) -> Result<QueryResult<'_, (Lrc<ast::Crate>, Rc<RefCell<BoxedResolver>>, Lrc<LintStore>)>>
    {
        trace!("expansion");
        self.expansion.compute(|| {
            let crate_name = *self.crate_name()?.borrow();
            let (krate, lint_store) = self.register_plugins()?.steal();
            let _timer = self.session().timer("configure_and_expand");
            let sess = self.session();
            let mut resolver = passes::create_resolver(
                sess.clone(),
                self.codegen_backend().metadata_loader(),
                &krate,
                crate_name,
            );
            let krate = resolver.access(|resolver| {
                passes::configure_and_expand(sess, &lint_store, krate, crate_name, resolver)
            })?;
            Ok((Lrc::new(krate), Rc::new(RefCell::new(resolver)), lint_store))
        })
    }

    fn dep_graph(&self) -> Result<QueryResult<'_, DepGraph>> {
        self.dep_graph.compute(|| {
            let sess = self.session();
            let future_opt = self.dep_graph_future()?.steal();
            let dep_graph = future_opt
                .and_then(|future| {
                    let (prev_graph, prev_work_products) =
                        sess.time("blocked_on_dep_graph_loading", || future.open().open(sess));

                    rustc_incremental::build_dep_graph(sess, prev_graph, prev_work_products)
                })
                .unwrap_or_else(DepGraph::new_disabled);
            Ok(dep_graph)
        })
    }

    pub fn global_ctxt(&'tcx self) -> Result<QueryResult<'_, &'tcx GlobalCtxt<'tcx>>> {
        self.gcx.compute(|| {
            let crate_name = *self.crate_name()?.borrow();
            let (krate, resolver, lint_store) = self.expansion()?.steal();

            let ty::ResolverOutputs {
                untracked,
                global_ctxt: untracked_resolutions,
                ast_lowering: untracked_resolver_for_lowering,
            } = BoxedResolver::to_resolver_outputs(resolver);

            let gcx = passes::create_global_ctxt(
                self.compiler,
                lint_store,
                self.dep_graph()?.steal(),
                untracked,
                &self.queries,
                &self.gcx_cell,
                &self.arena,
                &self.hir_arena,
            );

            gcx.enter(|tcx| {
                let feed = tcx.feed_unit_query();
                feed.resolver_for_lowering(
                    tcx.arena.alloc(Steal::new((untracked_resolver_for_lowering, krate))),
                );
                feed.resolutions(tcx.arena.alloc(untracked_resolutions));
                feed.features_query(tcx.sess.features_untracked());
                let feed = tcx.feed_local_crate();
                feed.crate_name(crate_name);
            });
            Ok(gcx)
        })
    }

    pub fn ongoing_codegen(&'tcx self) -> Result<QueryResult<'_, Box<dyn Any>>> {
        self.ongoing_codegen.compute(|| {
            self.global_ctxt()?.enter(|tcx| {
                tcx.analysis(()).ok();

                // Don't do code generation if there were any errors
                self.session().compile_status()?;

                // If we have any delayed bugs, for example because we created TyKind::Error earlier,
                // it's likely that codegen will only cause more ICEs, obscuring the original problem
                self.session().diagnostic().flush_delayed();

                // Hook for UI tests.
                Self::check_for_rustc_errors_attr(tcx);

                Ok(passes::start_codegen(&***self.codegen_backend(), tcx))
            })
        })
    }

    /// Check for the `#[rustc_error]` annotation, which forces an error in codegen. This is used
    /// to write UI tests that actually test that compilation succeeds without reporting
    /// an error.
    fn check_for_rustc_errors_attr(tcx: TyCtxt<'_>) {
        let Some((def_id, _)) = tcx.entry_fn(()) else { return };
        for attr in tcx.get_attrs(def_id, sym::rustc_error) {
            match attr.meta_item_list() {
                // Check if there is a `#[rustc_error(delay_span_bug_from_inside_query)]`.
                Some(list)
                    if list.iter().any(|list_item| {
                        matches!(
                            list_item.ident().map(|i| i.name),
                            Some(sym::delay_span_bug_from_inside_query)
                        )
                    }) =>
                {
                    tcx.ensure().trigger_delay_span_bug(def_id);
                }

                // Bare `#[rustc_error]`.
                None => {
                    tcx.sess.emit_fatal(RustcErrorFatal { span: tcx.def_span(def_id) });
                }

                // Some other attribute.
                Some(_) => {
                    tcx.sess.emit_warning(RustcErrorUnexpectedAnnotation {
                        span: tcx.def_span(def_id),
                    });
                }
            }
        }
    }

    pub fn linker(&'tcx self) -> Result<Linker> {
        let sess = self.session().clone();
        let codegen_backend = self.codegen_backend().clone();

        let (crate_hash, prepare_outputs, dep_graph) = self.global_ctxt()?.enter(|tcx| {
            (tcx.crate_hash(LOCAL_CRATE), tcx.output_filenames(()).clone(), tcx.dep_graph.clone())
        });
        let ongoing_codegen = self.ongoing_codegen()?.steal();

        Ok(Linker {
            sess,
            codegen_backend,

            dep_graph,
            prepare_outputs,
            crate_hash,
            ongoing_codegen,
        })
    }
}

pub struct Linker {
    // compilation inputs
    sess: Lrc<Session>,
    codegen_backend: Lrc<Box<dyn CodegenBackend>>,

    // compilation outputs
    dep_graph: DepGraph,
    prepare_outputs: Arc<OutputFilenames>,
    crate_hash: Svh,
    ongoing_codegen: Box<dyn Any>,
}

impl Linker {
    pub fn link(self) -> Result<()> {
        let (codegen_results, work_products) = self.codegen_backend.join_codegen(
            self.ongoing_codegen,
            &self.sess,
            &self.prepare_outputs,
        )?;

        self.sess.compile_status()?;

        let sess = &self.sess;
        let dep_graph = self.dep_graph;
        sess.time("serialize_work_products", || {
            rustc_incremental::save_work_product_index(sess, &dep_graph, work_products)
        });

        let prof = self.sess.prof.clone();
        prof.generic_activity("drop_dep_graph").run(move || drop(dep_graph));

        // Now that we won't touch anything in the incremental compilation directory
        // any more, we can finalize it (which involves renaming it)
        rustc_incremental::finalize_session_directory(&self.sess, self.crate_hash);

        if !self
            .sess
            .opts
            .output_types
            .keys()
            .any(|&i| i == OutputType::Exe || i == OutputType::Metadata)
        {
            return Ok(());
        }

        if sess.opts.unstable_opts.no_link {
            let encoded = CodegenResults::serialize_rlink(&codegen_results);
            let rlink_file = self.prepare_outputs.with_extension(config::RLINK_EXT);
            std::fs::write(&rlink_file, encoded)
                .map_err(|error| sess.emit_fatal(FailedWritingFile { path: &rlink_file, error }))?;
            return Ok(());
        }

        let _timer = sess.prof.verbose_generic_activity("link_crate");
        self.codegen_backend.link(&self.sess, codegen_results, &self.prepare_outputs)
    }
}

impl Compiler {
    pub fn enter<F, T>(&self, f: F) -> T
    where
        F: for<'tcx> FnOnce(&'tcx Queries<'tcx>) -> T,
    {
        let mut _timer = None;
        let queries = Queries::new(self);
        let ret = f(&queries);

        // NOTE: intentionally does not compute the global context if it hasn't been built yet,
        // since that likely means there was a parse error.
        if let Some(Ok(gcx)) = &mut *queries.gcx.result.borrow_mut() {
            let gcx = gcx.get_mut();
            // We assume that no queries are run past here. If there are new queries
            // after this point, they'll show up as "<unknown>" in self-profiling data.
            {
                let _prof_timer =
                    queries.session().prof.generic_activity("self_profile_alloc_query_strings");
                gcx.enter(rustc_query_impl::alloc_self_profile_query_strings);
            }

            self.session()
                .time("serialize_dep_graph", || gcx.enter(rustc_incremental::save_dep_graph));
        }

        _timer = Some(self.session().timer("free_global_ctxt"));

        ret
    }
}
