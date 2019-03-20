use crate::interface::{Compiler, Result};
use crate::passes::{self, BoxedResolver, ExpansionResult, BoxedGlobalCtxt};

use rustc_data_structures::sync::{Lrc, Lock, OneThread};
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
use std::sync::Arc;
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
    global_ctxt: Query<BoxedGlobalCtxt>,
    ongoing_codegen: Query<Lrc<ty::OngoingCodegen>>,
    link: Query<()>,
}

impl Compiler {
    pub fn global_ctxt(&self) -> Result<&Query<BoxedGlobalCtxt>> {
        self.queries.global_ctxt.compute(|| {
            Ok(passes::create_global_ctxt(
                self,
                self.io.clone(),
            ))
        })
    }

    pub fn ongoing_codegen(
        &self
    ) -> Result<&Query<Lrc<ty::OngoingCodegen>>> {
        self.queries.ongoing_codegen.compute(|| {
            self.global_ctxt()?.peek_mut().enter(|tcx| {
                tcx.ongoing_codegen(LOCAL_CRATE)
            })
        })
    }

    pub fn link(&self) -> Result<&Query<()>> {
        self.queries.link.compute(|| {
            let ongoing_codegen = self.ongoing_codegen()?.take();

            self.codegen_backend().join_codegen_and_link(
                OneThread::into_inner(ongoing_codegen.codegen_object.steal()),
                self.session(),
                &ongoing_codegen.dep_graph,
                &ongoing_codegen.outputs,
            ).map_err(|_| ErrorReported)?;

            Ok(())
        })
    }

    pub fn compile(&self) -> Result<()> {
        self.global_ctxt()?.peek_mut().enter(|tcx| {
            tcx.prepare_outputs(())?;
            Ok(())
        })?;

        if self.session().opts.output_types.contains_key(&OutputType::DepInfo)
            && self.session().opts.output_types.len() == 1
        {
            return Ok(())
        }

        // Drop AST after creating GlobalCtxt to free memory
        self.global_ctxt()?.peek_mut().enter(|tcx| {
            tcx.lower_ast_to_hir(())?;
            // Drop AST after lowering HIR to free memory
            mem::drop(tcx.expand_macros(()).unwrap().ast_crate.steal());
            Ok(())
        })?;

        self.ongoing_codegen()?;

        // Drop GlobalCtxt after starting codegen to free memory
        mem::drop(self.global_ctxt()?.take());

        self.link().map(|_| ())
    }
}
