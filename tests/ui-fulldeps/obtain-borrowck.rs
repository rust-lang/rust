//@ edition: 2021
//@ run-pass
//@ check-run-results
//@ run-flags: --sysroot {{sysroot-base}} --edition=2021 {{src-base}}/auxiliary/obtain-borrowck-input.rs
//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)
// ignore-tidy-linelength

#![feature(rustc_private)]

//! This program implements a rustc driver that retrieves MIR bodies with
//! borrowck information. This cannot be done in a straightforward way because
//! `get_bodies_with_borrowck_facts`–the function for retrieving MIR bodies with
//! borrowck facts–can panic if the bodies are stolen before it is invoked.
//! Therefore, the driver overrides `mir_borrowck` query (this is done in the
//! `config` callback), which retrieves the bodies that are about to be borrow
//! checked and stores them in a thread local `MIR_BODIES`. Then, `after_analysis`
//! callback triggers borrow checking of all MIR bodies by retrieving
//! `optimized_mir` and pulls out the MIR bodies with the borrowck information
//! from the thread local storage.

extern crate rustc_borrowck;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;

use std::cell::RefCell;
use std::collections::HashMap;
use std::thread_local;

use rustc_borrowck::consumers::{self, BodyWithBorrowckFacts, ConsumerOptions};
use rustc_data_structures::fx::FxHashMap;
use rustc_driver::Compilation;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_interface::Config;
use rustc_interface::interface::Compiler;
use rustc_middle::query::queries::mir_borrowck::ProvidedValue;
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;

fn main() {
    let exit_code = rustc_driver::catch_with_exit_code(move || {
        let mut rustc_args: Vec<_> = std::env::args().collect();
        // We must pass -Zpolonius so that the borrowck information is computed.
        rustc_args.push("-Zpolonius".to_owned());
        let mut callbacks = CompilerCalls::default();
        // Call the Rust compiler with our callbacks.
        rustc_driver::run_compiler(&rustc_args, &mut callbacks);
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
    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        tcx.sess.dcx().abort_if_errors();
        // Collect definition ids of MIR bodies.
        let mut bodies = Vec::new();

        let crate_items = tcx.hir_crate_items(());
        for id in crate_items.free_items() {
            if matches!(tcx.def_kind(id.owner_id), DefKind::Fn) {
                bodies.push(id.owner_id);
            }
        }

        for id in crate_items.trait_items() {
            if matches!(tcx.def_kind(id.owner_id), DefKind::AssocFn) {
                let trait_item = tcx.hir_trait_item(id);
                if let rustc_hir::TraitItemKind::Fn(_, trait_fn) = &trait_item.kind {
                    if let rustc_hir::TraitFn::Provided(_) = trait_fn {
                        bodies.push(trait_item.owner_id);
                    }
                }
            }
        }

        for id in crate_items.impl_items() {
            if matches!(tcx.def_kind(id.owner_id), DefKind::AssocFn) {
                bodies.push(id.owner_id);
            }
        }

        // Trigger borrow checking of all bodies.
        for def_id in bodies {
            let _ = tcx.optimized_mir(def_id);
        }

        // See what bodies were borrow checked.
        let mut bodies = get_bodies(tcx);
        bodies.sort_by(|(def_id1, _), (def_id2, _)| def_id1.cmp(def_id2));
        println!("Bodies retrieved for:");
        for (def_id, body) in bodies {
            println!("{}", def_id);
            assert!(body.input_facts.unwrap().cfg_edge.len() > 0);
        }

        Compilation::Continue
    }
}

fn override_queries(_session: &Session, local: &mut Providers) {
    local.mir_borrowck = mir_borrowck;
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

fn mir_borrowck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> ProvidedValue<'tcx> {
    let opts = ConsumerOptions::PoloniusInputFacts;
    let bodies_with_facts = consumers::get_bodies_with_borrowck_facts(tcx, def_id, opts);
    // SAFETY: The reader casts the 'static lifetime to 'tcx before using it.
    let bodies_with_facts: FxHashMap<LocalDefId, BodyWithBorrowckFacts<'static>> =
        unsafe { std::mem::transmute(bodies_with_facts) };
    MIR_BODIES.with(|state| {
        let mut map = state.borrow_mut();
        for (def_id, body_with_facts) in bodies_with_facts {
            assert!(map.insert(def_id, body_with_facts).is_none());
        }
    });
    let mut providers = Providers::default();
    rustc_borrowck::provide(&mut providers);
    let original_mir_borrowck = providers.mir_borrowck;
    original_mir_borrowck(tcx, def_id)
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
                (def_path.to_string_no_crate_verbose(), unsafe { std::mem::transmute(body) })
            })
            .collect()
    })
}
