use crate::errors::{FailedWritingFile, RustcErrorFatal, RustcErrorUnexpectedAnnotation};
use crate::interface::{Compiler, Result};
use crate::{passes, util};

use rustc_ast as ast;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::CodegenResults;
use rustc_data_structures::steal::Steal;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{AppendOnlyIndexVec, FreezeLock, OnceLock, WorkerLocal};
use rustc_hir::def_id::{StableCrateId, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::definitions::Definitions;
use rustc_incremental::setup_dep_graph;
use rustc_metadata::creader::CStore;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::DepGraph;
use rustc_middle::ty::{GlobalCtxt, TyCtxt};
use rustc_session::config::{self, CrateType, OutputFilenames, OutputType};
use rustc_session::cstore::Untracked;
use rustc_session::{output::find_crate_name, Session};
use rustc_span::symbol::sym;
use std::any::Any;
use std::cell::{RefCell, RefMut};
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
    gcx_cell: OnceLock<GlobalCtxt<'tcx>>,

    arena: WorkerLocal<Arena<'tcx>>,
    hir_arena: WorkerLocal<rustc_hir::Arena<'tcx>>,

    parse: Query<ast::Crate>,
    pre_configure: Query<(ast::Crate, ast::AttrVec)>,
    // This just points to what's in `gcx_cell`.
    gcx: Query<&'tcx GlobalCtxt<'tcx>>,
}

impl<'tcx> Queries<'tcx> {
    pub fn new(compiler: &'tcx Compiler) -> Queries<'tcx> {
        Queries {
            compiler,
            gcx_cell: OnceLock::new(),
            arena: WorkerLocal::new(|_| Arena::default()),
            hir_arena: WorkerLocal::new(|_| rustc_hir::Arena::default()),
            parse: Default::default(),
            pre_configure: Default::default(),
            gcx: Default::default(),
        }
    }

    fn session(&self) -> &Session {
        &self.compiler.session()
    }

    fn codegen_backend(&self) -> &dyn CodegenBackend {
        self.compiler.codegen_backend()
    }

    pub fn parse(&self) -> Result<QueryResult<'_, ast::Crate>> {
        self.parse
            .compute(|| passes::parse(self.session()).map_err(|mut parse_error| parse_error.emit()))
    }

    #[deprecated = "pre_configure may be made private in the future. If you need it please open an issue with your use case."]
    pub fn pre_configure(&self) -> Result<QueryResult<'_, (ast::Crate, ast::AttrVec)>> {
        self.pre_configure.compute(|| {
            let mut krate = self.parse()?.steal();

            let sess = self.session();
            rustc_builtin_macros::cmdline_attrs::inject(
                &mut krate,
                &sess.parse_sess,
                &sess.opts.unstable_opts.crate_attr,
            );

            let pre_configured_attrs =
                rustc_expand::config::pre_configure_attrs(sess, &krate.attrs);
            Ok((krate, pre_configured_attrs))
        })
    }

    pub fn global_ctxt(&'tcx self) -> Result<QueryResult<'_, &'tcx GlobalCtxt<'tcx>>> {
        self.gcx.compute(|| {
            let sess = self.session();
            #[allow(deprecated)]
            let (krate, pre_configured_attrs) = self.pre_configure()?.steal();

            // parse `#[crate_name]` even if `--crate-name` was passed, to make sure it matches.
            let crate_name = find_crate_name(sess, &pre_configured_attrs);
            let crate_types = util::collect_crate_types(sess, &pre_configured_attrs);
            let stable_crate_id = StableCrateId::new(
                crate_name,
                crate_types.contains(&CrateType::Executable),
                sess.opts.cg.metadata.clone(),
                sess.cfg_version,
            );
            let dep_graph = setup_dep_graph(sess, crate_name, stable_crate_id)?;

            let cstore = FreezeLock::new(Box::new(CStore::new(
                self.codegen_backend().metadata_loader(),
                stable_crate_id,
            )) as _);
            let definitions = FreezeLock::new(Definitions::new(stable_crate_id));
            let source_span = AppendOnlyIndexVec::new();
            let _id = source_span.push(krate.spans.inner_span);
            debug_assert_eq!(_id, CRATE_DEF_ID);
            let untracked = Untracked { cstore, source_span, definitions };

            let qcx = passes::create_global_ctxt(
                self.compiler,
                crate_types,
                stable_crate_id,
                dep_graph,
                untracked,
                &self.gcx_cell,
                &self.arena,
                &self.hir_arena,
            );

            qcx.enter(|tcx| {
                let feed = tcx.feed_local_crate();
                feed.crate_name(crate_name);

                let feed = tcx.feed_unit_query();
                feed.features_query(tcx.arena.alloc(rustc_expand::config::features(
                    sess,
                    &pre_configured_attrs,
                    crate_name,
                )));
                feed.crate_for_resolver(tcx.arena.alloc(Steal::new((krate, pre_configured_attrs))));
            });
            Ok(qcx)
        })
    }

    pub fn ongoing_codegen(&'tcx self) -> Result<Box<dyn Any>> {
        self.global_ctxt()?.enter(|tcx| {
            // Don't do code generation if there were any errors
            self.session().compile_status()?;

            // If we have any delayed bugs, for example because we created TyKind::Error earlier,
            // it's likely that codegen will only cause more ICEs, obscuring the original problem
            self.session().diagnostic().flush_delayed();

            // Hook for UI tests.
            Self::check_for_rustc_errors_attr(tcx);

            Ok(passes::start_codegen(self.codegen_backend(), tcx))
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

    pub fn linker(&'tcx self, ongoing_codegen: Box<dyn Any>) -> Result<Linker> {
        self.global_ctxt()?.enter(|tcx| {
            Ok(Linker {
                dep_graph: tcx.dep_graph.clone(),
                output_filenames: tcx.output_filenames(()).clone(),
                crate_hash: if tcx.needs_crate_hash() {
                    Some(tcx.crate_hash(LOCAL_CRATE))
                } else {
                    None
                },
                ongoing_codegen,
            })
        })
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
    pub fn link(self, sess: &Session, codegen_backend: &dyn CodegenBackend) -> Result<()> {
        let (codegen_results, work_products) =
            codegen_backend.join_codegen(self.ongoing_codegen, sess, &self.output_filenames)?;

        sess.compile_status()?;

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
            return Ok(());
        }

        if sess.opts.unstable_opts.no_link {
            let rlink_file = self.output_filenames.with_extension(config::RLINK_EXT);
            CodegenResults::serialize_rlink(sess, &rlink_file, &codegen_results)
                .map_err(|error| sess.emit_fatal(FailedWritingFile { path: &rlink_file, error }))?;
            return Ok(());
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
