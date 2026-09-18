//@ edition: 2021
//@ run-pass
//@ check-run-results
// ignore-tidy-linelength
//@ run-flags: --sysroot {{sysroot-base}} --edition=2021 {{src-base}}/auxiliary/obtain-borrowck-input.rs
//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)

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

use std::collections::HashMap;
use std::process::ExitCode;
use std::sync::{LazyLock, Mutex};

use rustc_borrowck::consumers::{self, ConsumerOptions, PoloniusInput};
use rustc_driver::Compilation;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_interface::Config;
use rustc_interface::interface::Compiler;
use rustc_middle::queries::mir_borrowck::ProvidedValue;
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;

fn main() -> ExitCode {
    rustc_driver::catch_with_exit_code(move || {
        let mut rustc_args: Vec<_> = std::env::args().collect();
        // We must pass -Zpolonius so that the borrowck information is computed.
        rustc_args.push("-Zpolonius".to_owned());
        let mut callbacks = CompilerCalls::default();
        // Call the Rust compiler with our callbacks.
        rustc_driver::run_compiler(&rustc_args, &mut callbacks);
    })
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
        for (def_id, facts) in bodies {
            println!("{}", def_id);
            assert!(facts.cfg_edge.len() > 0);
        }

        Compilation::Continue
    }
}

fn override_queries(_session: &Session, local: &mut Providers) {
    local.queries.mir_borrowck = mir_borrowck;
}

// Since mir_borrowck does not have access to any other state, we need to use a
// global variable for storing the obtained MIR bodies.
// Note that we don't use a thread-local variable, because borrowck can run under multiple threads.
pub static MIR_BODIES: LazyLock<Mutex<HashMap<LocalDefId, Option<Box<PoloniusInput>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn mir_borrowck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> ProvidedValue<'tcx> {
    let opts = ConsumerOptions::PoloniusInputFacts;
    let bodies_with_facts = consumers::get_bodies_with_borrowck_facts(tcx, def_id, opts);
    let mut map = MIR_BODIES.lock().unwrap();
    for (def_id, body_with_facts) in bodies_with_facts {
        assert!(map.insert(def_id, body_with_facts.input_facts).is_none());
    }
    let mut providers = Providers::default();
    rustc_borrowck::provide(&mut providers.queries);
    let original_mir_borrowck = providers.queries.mir_borrowck;
    original_mir_borrowck(tcx, def_id)
}

/// Pull MIR bodies stored in the global variable.
fn get_bodies<'tcx>(tcx: TyCtxt<'tcx>) -> Vec<(String, PoloniusInput)> {
    let mut map = MIR_BODIES.lock().unwrap();
    map.drain()
        .map(|(def_id, facts)| {
            let def_path = tcx.def_path(def_id.to_def_id());
            (def_path.to_string_no_crate_verbose(), *facts.unwrap())
        })
        .collect()
}
