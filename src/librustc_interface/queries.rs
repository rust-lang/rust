use crate::interface::{Compiler, Result};
use crate::passes::{self, BoxedResolver, ExpansionResult, BoxedGlobalCtxt, PluginInfo};

use rustc_incremental::DepGraphFuture;
use rustc_data_structures::sync::Lrc;
use rustc::session::config::{Input, OutputFilenames, OutputType};
use rustc::session::Session;
use rustc::util::common::{time, ErrorReported};
use rustc::util::profiling::ProfileCategory;
use rustc::lint;
use rustc::hir;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::ty;
use rustc::ty::steal::Steal;
use rustc::dep_graph::DepGraph;
use rustc_passes::hir_stats;
use rustc_plugin::registry::Registry;
use serialize::json;
use std::cell::{Ref, RefMut, RefCell};
use std::ops::Deref;
use std::rc::Rc;
use std::sync::mpsc;
use std::any::Any;
use std::mem;
use syntax::parse::{self, PResult};
use syntax::util::node_count::NodeCounter;
use syntax::{self, ast, attr, diagnostics, visit};
use syntax_pos::hygiene;

/// Represent the result of a query.
/// This result can be stolen with the `take` method and returned with the `give` method.
pub struct Query<T> {
    result: RefCell<Option<Result<T>>>,
}

impl<T> Query<T> {
    fn compute<F: FnOnce() -> Result<T>>(&self, f: F) -> Result<&Query<T>> {
        let mut result = self.result.borrow_mut();
        if result.is_none() {
            *result = Some(f());
        }
        result.as_ref().unwrap().as_ref().map(|_| self).map_err(|err| *err)
    }

    /// Takes ownership of the query result. Further attempts to take or peek the query
    /// result will panic unless it is returned by calling the `give` method.
    pub fn take(&self) -> T {
        self.result
            .borrow_mut()
            .take()
            .expect("missing query result")
            .unwrap()
    }

    /// Returns a stolen query result. Panics if there's already a result.
    pub fn give(&self, value: T) {
        let mut result = self.result.borrow_mut();
        assert!(result.is_none(), "a result already exists");
        *result = Some(Ok(value));
    }

    /// Borrows the query result using the RefCell. Panics if the result is stolen.
    pub fn peek(&self) -> Ref<'_, T> {
        Ref::map(self.result.borrow(), |r| {
            r.as_ref().unwrap().as_ref().expect("missing query result")
        })
    }

    /// Mutably borrows the query result using the RefCell. Panics if the result is stolen.
    pub fn peek_mut(&self) -> RefMut<'_, T> {
        RefMut::map(self.result.borrow_mut(), |r| {
            r.as_mut().unwrap().as_mut().expect("missing query result")
        })
    }
}

impl<T> Default for Query<T> {
    fn default() -> Self {
        Query {
            result: RefCell::new(None),
        }
    }
}

#[derive(Default)]
pub(crate) struct Queries {
    dep_graph_future: Query<Option<DepGraphFuture>>,
    parse: Query<ast::Crate>,
    crate_name: Query<String>,
    register_plugins: Query<(ast::Crate, PluginInfo)>,
    expansion: Query<(ast::Crate, Rc<Option<RefCell<BoxedResolver>>>)>,
    dep_graph: Query<DepGraph>,
    lower_to_hir: Query<(Steal<hir::map::Forest>, ExpansionResult)>,
    prepare_outputs: Query<OutputFilenames>,
    codegen_channel: Query<(Steal<mpsc::Sender<Box<dyn Any + Send>>>,
                            Steal<mpsc::Receiver<Box<dyn Any + Send>>>)>,
    global_ctxt: Query<BoxedGlobalCtxt>,
    ongoing_codegen: Query<Box<dyn Any>>,
    link: Query<()>,
}

impl Compiler {
    pub fn dep_graph_future(&self) -> Result<&Query<Option<DepGraphFuture>>> {
        self.queries.dep_graph_future.compute(|| {
            Ok(if self.session().opts.build_dep_graph() {
                Some(rustc_incremental::load_dep_graph(self.session()))
            } else {
                None
            })
        })
    }

    pub fn parse(&self) -> Result<&Query<ast::Crate>> {
        self.queries.parse.compute(|| {
            passes::parse(self.session(), &self.input).map_err(
                |mut parse_error| {
                    parse_error.emit();
                    ErrorReported
                },
            )
        })
    }

    pub fn register_plugins(&self) -> Result<&Query<(ast::Crate, PluginInfo)>> {
        self.queries.register_plugins.compute(|| {
            let crate_name = self.crate_name()?.peek().clone();
            let krate = self.parse()?.take();

            passes::register_plugins(
                self,
                self.session(),
                self.cstore(),
                krate,
                &crate_name,
            )
        })
    }

    pub fn crate_name(&self) -> Result<&Query<String>> {
        self.queries.crate_name.compute(|| {
            let parse_result = self.parse()?;
            let krate = parse_result.peek();
            let result = match self.crate_name {
                Some(ref crate_name) => crate_name.clone(),
                None => rustc_codegen_utils::link::find_crate_name(
                    Some(self.session()),
                    &krate.attrs,
                    &self.input
                ),
            };
            Ok(result)
        })
    }

    pub fn expansion(
        &self
    ) -> Result<&Query<(ast::Crate, Rc<Option<RefCell<BoxedResolver>>>)>> {
        self.queries.expansion.compute(|| {
            let crate_name = self.crate_name()?.peek().clone();
            let (krate, plugin_info) = self.register_plugins()?.take();
            passes::configure_and_expand(
                self.sess.clone(),
                self.cstore().clone(),
                krate,
                &crate_name,
                plugin_info,
            ).map(|(krate, resolver)| (krate, Rc::new(Some(RefCell::new(resolver)))))
        })
    }

    pub fn dep_graph(&self) -> Result<&Query<DepGraph>> {
        self.queries.dep_graph.compute(|| {
            Ok(match self.dep_graph_future()?.take() {
                None => DepGraph::new_disabled(),
                Some(future) => {
                    let (prev_graph, prev_work_products) =
                        time(self.session(), "blocked while dep-graph loading finishes", || {
                            future.open().unwrap_or_else(|e| rustc_incremental::LoadResult::Error {
                                message: format!("could not decode incremental cache: {:?}", e),
                            }).open(self.session())
                        });
                    DepGraph::new(prev_graph, prev_work_products)
                }
            })
        })
    }

    pub fn lower_to_hir(&self) -> Result<&Query<(Steal<hir::map::Forest>, ExpansionResult)>> {
        self.queries.lower_to_hir.compute(|| {
            let expansion_result = self.expansion()?;
            let (krate, resolver) = expansion_result.take();
            let resolver_ref = &*resolver;
            let hir = Steal::new(resolver_ref.as_ref().unwrap().borrow_mut().access(|resolver| {
                passes::lower_to_hir(
                    self.session(),
                    self.cstore(),
                    resolver,
                    &*self.dep_graph()?.peek(),
                    &krate
                )
            })?);
            expansion_result.give((krate, Rc::new(None)));
            Ok((hir, BoxedResolver::to_expansion_result(resolver)))
        })
    }

    pub fn prepare_outputs(&self) -> Result<&Query<OutputFilenames>> {
        self.queries.prepare_outputs.compute(|| {
            self.lower_to_hir()?;
            let krate = self.expansion()?;
            let krate = krate.peek();
            let crate_name = self.crate_name()?;
            let crate_name = crate_name.peek();
            passes::prepare_outputs(self.session(), self, &krate.0, &*crate_name)
        })
    }

    pub fn codegen_channel(&self) -> Result<&Query<(Steal<mpsc::Sender<Box<dyn Any + Send>>>,
                                                    Steal<mpsc::Receiver<Box<dyn Any + Send>>>)>> {
        self.queries.codegen_channel.compute(|| {
            let (tx, rx) = mpsc::channel();
            Ok((Steal::new(tx), Steal::new(rx)))
        })
    }

    pub fn global_ctxt(&self) -> Result<&Query<BoxedGlobalCtxt>> {
        self.queries.global_ctxt.compute(|| {
            let crate_name = self.crate_name()?.peek().clone();
            let outputs = self.prepare_outputs()?.peek().clone();
            let hir = self.lower_to_hir()?;
            let hir = hir.peek();
            let (ref hir_forest, ref expansion) = *hir;
            let tx = self.codegen_channel()?.peek().0.steal();
            Ok(passes::create_global_ctxt(
                self,
                hir_forest.steal(),
                expansion.defs.steal(),
                expansion.resolutions.steal(),
                outputs,
                tx,
                &crate_name))
        })
    }

    pub fn ongoing_codegen(&self) -> Result<&Query<Box<dyn Any>>> {
        self.queries.ongoing_codegen.compute(|| {
            let rx = self.codegen_channel()?.peek().1.steal();
            let outputs = self.prepare_outputs()?;
            self.global_ctxt()?.peek_mut().enter(|tcx| {
                tcx.analysis(LOCAL_CRATE).ok();

                // Don't do code generation if there were any errors
                self.session().compile_status()?;

                Ok(passes::start_codegen(
                    &***self.codegen_backend(),
                    tcx,
                    rx,
                    &*outputs.peek()
                ))
            })
        })
    }

    pub fn link(&self) -> Result<&Query<()>> {
        self.queries.link.compute(|| {
            let sess = self.session();

            let ongoing_codegen = self.ongoing_codegen()?.take();

            self.codegen_backend().join_codegen_and_link(
                ongoing_codegen,
                sess,
                &*self.dep_graph()?.peek(),
                &*self.prepare_outputs()?.peek(),
            ).map_err(|_| ErrorReported)?;

            Ok(())
        })
    }

    pub fn compile(&self) -> Result<()> {
        self.prepare_outputs()?;

        if self.session().opts.output_types.contains_key(&OutputType::DepInfo)
            && self.session().opts.output_types.len() == 1
        {
            return Ok(())
        }

        self.global_ctxt()?;

        // Drop AST after creating GlobalCtxt to free memory
        mem::drop(self.expansion()?.take());

        self.ongoing_codegen()?;

        // Drop GlobalCtxt after starting codegen to free memory
        mem::drop(self.global_ctxt()?.take());

        self.link().map(|_| ())
    }
}
