#![feature(rustc_private)]

//! This program implements a rustc driver that retrieves MIR bodies with
//! borrowck information. This cannot be done in a straightforward way because
//! `get_body_with_borrowck_facts`–the function for retrieving a MIR body with
//! borrowck facts–can panic if the body is stolen before it is invoked.
//! Therefore, the driver overrides `mir_borrowck` query (this is done in the
//! `config` callback), which retrieves the body that is about to be borrow
//! checked and stores it in a thread local `MIR_BODIES`. Then, `after_analysis`
//! callback triggers borrow checking of all MIR bodies by retrieving
//! `optimized_mir` and pulls out the MIR bodies with the borrowck information
//! from the thread local storage.

extern crate rustc_borrowck;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;

use rustc_borrowck::consumers::BodyWithBorrowckFacts;
use rustc_driver::Compilation;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_interface::interface::Compiler;
use rustc_interface::{Config, Queries};
use rustc_middle::ty::query::query_values::mir_borrowck;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::Session;
use std::cell::RefCell;
use std::collections::HashMap;
use std::thread_local;

fn main() {
    let exit_code = rustc_driver::catch_with_exit_code(move || {
        let mut rustc_args: Vec<_> = std::env::args().collect();
        // We must pass -Zpolonius so that the borrowck information is computed.
        rustc_args.push("-Zpolonius".to_owned());
        let mut callbacks = CompilerCalls::default();
        // Call the Rust compiler with our callbacks.
        rustc_driver::RunCompiler::new(&rustc_args, &mut callbacks).run()
    });
    std::process::exit(exit_code);
}

#[derive(Default)]
pub struct CompilerCalls;

impl rustc_driver::Callbacks for CompilerCalls {

    // In this callback we override the mir_borrowck query.
    fn config(&mut self, config: &mut Config) {
        assert!(config.override_queries.is_none());
        config.override_queries = Some(override_queries);
    }

    // In this callback we trigger borrow checking of all functions and obtain
    // the result.
    fn after_analysis<'tcx>(
        &mut self,
        compiler: &Compiler,
        queries: &'tcx Queries<'tcx>,
    ) -> Compilation {
        compiler.session().abort_if_errors();
        queries.global_ctxt().unwrap().peek_mut().enter(|tcx| {

            // Collect definition ids of MIR bodies.
            let hir = tcx.hir();
            let krate = hir.krate();
            let mut visitor = HirVisitor { bodies: Vec::new() };
            krate.visit_all_item_likes(&mut visitor);

            // Trigger borrow checking of all bodies.
            for def_id in visitor.bodies {
                let _ = tcx.optimized_mir(def_id);
            }

            // See what bodies were borrow checked.
            let mut bodies = get_bodies(tcx);
            bodies.sort_by(|(def_id1, _), (def_id2, _)| def_id1.cmp(def_id2));
            println!("Bodies retrieved for:");
            for (def_id, body) in bodies {
                println!("{}", def_id);
                assert!(body.input_facts.cfg_edge.len() > 0);
            }
        });

        Compilation::Continue
    }
}

fn override_queries(_session: &Session, local: &mut Providers, external: &mut Providers) {
    local.mir_borrowck = mir_borrowck;
    external.mir_borrowck = mir_borrowck;
}

// Since mir_borrowck does not have access to any other state, we need to use a
// thread-local for storing the obtained MIR bodies.
//
// Note: We are using 'static lifetime here, which is in general unsound.
// Unfortunately, that is the only lifetime allowed here. Our use is safe
// because we cast it back to `'tcx` before using.
thread_local! {
    pub static MIR_BODIES:
        RefCell<HashMap<LocalDefId, BodyWithBorrowckFacts<'static>>> =
        RefCell::new(HashMap::new());
}

fn mir_borrowck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> mir_borrowck<'tcx> {
    let body_with_facts = rustc_borrowck::consumers::get_body_with_borrowck_facts(
        tcx,
        ty::WithOptConstParam::unknown(def_id),
    );
    // SAFETY: The reader casts the 'static lifetime to 'tcx before using it.
    let body_with_facts: BodyWithBorrowckFacts<'static> =
        unsafe { std::mem::transmute(body_with_facts) };
    MIR_BODIES.with(|state| {
        let mut map = state.borrow_mut();
        assert!(map.insert(def_id, body_with_facts).is_none());
    });
    let mut providers = Providers::default();
    rustc_borrowck::provide(&mut providers);
    let original_mir_borrowck = providers.mir_borrowck;
    original_mir_borrowck(tcx, def_id)
}

/// Visitor that collects all body definition ids mentioned in the program.
struct HirVisitor {
    bodies: Vec<LocalDefId>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for HirVisitor {
    fn visit_item(&mut self, item: &rustc_hir::Item) {
        if let rustc_hir::ItemKind::Fn(..) = item.kind {
            self.bodies.push(item.def_id);
        }
    }

    fn visit_trait_item(&mut self, trait_item: &rustc_hir::TraitItem) {
        if let rustc_hir::TraitItemKind::Fn(_, trait_fn) = &trait_item.kind {
            if let rustc_hir::TraitFn::Provided(_) = trait_fn {
                self.bodies.push(trait_item.def_id);
            }
        }
    }

    fn visit_impl_item(&mut self, impl_item: &rustc_hir::ImplItem) {
        if let rustc_hir::ImplItemKind::Fn(..) = impl_item.kind {
            self.bodies.push(impl_item.def_id);
        }
    }

    fn visit_foreign_item(&mut self, _foreign_item: &rustc_hir::ForeignItem) {}
}

/// Pull MIR bodies stored in the thread-local.
fn get_bodies<'tcx>(tcx: TyCtxt<'tcx>) -> Vec<(String, BodyWithBorrowckFacts<'tcx>)> {
    MIR_BODIES.with(|state| {
        let mut map = state.borrow_mut();
        map.drain()
            .map(|(def_id, body)| {
                let def_path = tcx.def_path(def_id.to_def_id());
                // SAFETY: For soundness we need to ensure that the bodies have
                // the same lifetime (`'tcx`), which they had before they were
                // stored in the thread local.
                (def_path.to_string_no_crate_verbose(), body)
            })
            .collect()
    })
}
