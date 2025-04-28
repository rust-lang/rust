// Tested with nightly-2025-03-28

#![feature(rustc_private)]

extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_error_codes;
extern crate rustc_errors;
extern crate rustc_hash;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use std::sync::{Arc, Mutex};

use rustc_errors::emitter::Emitter;
use rustc_errors::registry::{self, Registry};
use rustc_errors::translation::Translate;
use rustc_errors::{DiagInner, FluentBundle};
use rustc_session::config;
use rustc_span::source_map::SourceMap;

struct DebugEmitter {
    source_map: Arc<SourceMap>,
    diagnostics: Arc<Mutex<Vec<DiagInner>>>,
}

impl Translate for DebugEmitter {
    fn fluent_bundle(&self) -> Option<&FluentBundle> {
        None
    }

    fn fallback_fluent_bundle(&self) -> &FluentBundle {
        panic!("this emitter should not translate message")
    }
}

impl Emitter for DebugEmitter {
    fn emit_diagnostic(&mut self, diag: DiagInner, _: &Registry) {
        self.diagnostics.lock().unwrap().push(diag);
    }

    fn source_map(&self) -> Option<&SourceMap> {
        Some(&self.source_map)
    }
}

fn main() {
    let buffer: Arc<Mutex<Vec<DiagInner>>> = Arc::default();
    let diagnostics = buffer.clone();
    let config = rustc_interface::Config {
        opts: config::Options::default(),
        // This program contains a type error.
        input: config::Input::Str {
            name: rustc_span::FileName::Custom("main.rs".into()),
            input: "
fn main() {
    let x: &str = 1;
}
"
            .into(),
        },
        crate_cfg: Vec::new(),
        crate_check_cfg: Vec::new(),
        output_dir: None,
        output_file: None,
        file_loader: None,
        locale_resources: rustc_driver::DEFAULT_LOCALE_RESOURCES.to_owned(),
        lint_caps: rustc_hash::FxHashMap::default(),
        psess_created: Some(Box::new(|parse_sess| {
            parse_sess.dcx().set_emitter(Box::new(DebugEmitter {
                source_map: parse_sess.clone_source_map(),
                diagnostics,
            }));
        })),
        register_lints: None,
        override_queries: None,
        registry: registry::Registry::new(rustc_errors::codes::DIAGNOSTICS),
        make_codegen_backend: None,
        expanded_args: Vec::new(),
        ice_file: None,
        hash_untracked_state: None,
        using_internal_features: &rustc_driver::USING_INTERNAL_FEATURES,
    };
    rustc_interface::run_compiler(config, |compiler| {
        let krate = rustc_interface::passes::parse(&compiler.sess);
        rustc_interface::create_and_enter_global_ctxt(&compiler, krate, |tcx| {
            // Iterate all the items defined and perform type checking.
            tcx.par_hir_body_owners(|item_def_id| {
                tcx.ensure_ok().typeck(item_def_id);
            });
        });
        // If the compiler has encountered errors when this closure returns, it will abort (!) the program.
        // We avoid this by resetting the error count before returning
        compiler.sess.dcx().reset_err_count();
    });
    // Read buffered diagnostics.
    buffer.lock().unwrap().iter().for_each(|diagnostic| {
        println!("{diagnostic:#?}");
    });
}
