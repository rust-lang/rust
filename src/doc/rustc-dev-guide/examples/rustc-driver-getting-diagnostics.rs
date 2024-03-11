#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_error_codes;
extern crate rustc_errors;
extern crate rustc_hash;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use rustc_errors::{
    emitter::Emitter, registry, translation::Translate, DiagCtxt, DiagInner, FluentBundle,
};
use rustc_session::config;
use rustc_span::source_map::SourceMap;

use std::{
    path, process, str,
    sync::{Arc, Mutex},
};

struct DebugEmitter {
    source_map: Arc<SourceMap>,
    diagnostics: Arc<Mutex<Vec<DiagInner>>>,
}

impl Translate for DebugEmitter {
    fn fluent_bundle(&self) -> Option<&Arc<FluentBundle>> {
        None
    }

    fn fallback_fluent_bundle(&self) -> &FluentBundle {
        panic!("this emitter should not translate message")
    }
}

impl Emitter for DebugEmitter {
    fn emit_diagnostic(&mut self, diag: &DiagInner) {
        self.diagnostics.lock().unwrap().push(diag.clone());
    }

    fn source_map(&self) -> Option<&Arc<SourceMap>> {
        Some(&self.source_map)
    }
}

fn main() {
    let out = process::Command::new("rustc")
        .arg("--print=sysroot")
        .current_dir(".")
        .output()
        .unwrap();
    let sysroot = str::from_utf8(&out.stdout).unwrap().trim();
    let buffer: Arc<Mutex<Vec<DiagInner>>> = Arc::default();
    let diagnostics = buffer.clone();
    let config = rustc_interface::Config {
        opts: config::Options {
            maybe_sysroot: Some(path::PathBuf::from(sysroot)),
            ..config::Options::default()
        },
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
        locale_resources: rustc_driver::DEFAULT_LOCALE_RESOURCES,
        lint_caps: rustc_hash::FxHashMap::default(),
        parse_sess_created: Some(Box::new(|parse_sess| {
            parse_sess.dcx = DiagCtxt::with_emitter(Box::new(DebugEmitter {
                source_map: parse_sess.clone_source_map(),
                diagnostics,
            }))
        })),
        register_lints: None,
        override_queries: None,
        registry: registry::Registry::new(rustc_error_codes::DIAGNOSTICS),
        make_codegen_backend: None,
        expanded_args: Vec::new(),
        ice_file: None,
        hash_untracked_state: None,
        using_internal_features: Arc::default(),
    };
    rustc_interface::run_compiler(config, |compiler| {
        compiler.enter(|queries| {
            queries.global_ctxt().unwrap().enter(|tcx| {
                // Run the analysis phase on the local crate to trigger the type error.
                let _ = tcx.analysis(());
            });
        });
    });
    // Read buffered diagnostics.
    buffer.lock().unwrap().iter().for_each(|diagnostic| {
        println!("{diagnostic:#?}");
    });
}
