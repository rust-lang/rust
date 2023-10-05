#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_error_codes;
extern crate rustc_errors;
extern crate rustc_hash;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use rustc_errors::registry;
use rustc_session::config::{self, CheckCfg};
use rustc_span::source_map;
use std::io;
use std::path;
use std::process;
use std::str;
use std::sync;

// Buffer diagnostics in a Vec<u8>.
#[derive(Clone)]
pub struct DiagnosticSink(sync::Arc<sync::Mutex<Vec<u8>>>);

impl io::Write for DiagnosticSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

fn main() {
    let out = process::Command::new("rustc")
        .arg("--print=sysroot")
        .current_dir(".")
        .output()
        .unwrap();
    let sysroot = str::from_utf8(&out.stdout).unwrap().trim();
    let buffer = sync::Arc::new(sync::Mutex::new(Vec::new()));
    let config = rustc_interface::Config {
        opts: config::Options {
            maybe_sysroot: Some(path::PathBuf::from(sysroot)),
            // Configure the compiler to emit diagnostics in compact JSON format.
            error_format: config::ErrorOutputType::Json {
                pretty: false,
                json_rendered: rustc_errors::emitter::HumanReadableErrorType::Default(
                    rustc_errors::emitter::ColorConfig::Never,
                ),
            },
            ..config::Options::default()
        },
        // This program contains a type error.
        input: config::Input::Str {
            name: source_map::FileName::Custom("main.rs".into()),
            input: "
fn main() {
    let x: &str = 1;
}
"
            .into(),
        },
        crate_cfg: rustc_hash::FxHashSet::default(),
        crate_check_cfg: CheckCfg::default(),
        output_dir: None,
        output_file: None,
        file_loader: None,
        locale_resources: rustc_driver::DEFAULT_LOCALE_RESOURCES,
        lint_caps: rustc_hash::FxHashMap::default(),
        parse_sess_created: None,
        register_lints: None,
        override_queries: None,
        registry: registry::Registry::new(&rustc_error_codes::DIAGNOSTICS),
        make_codegen_backend: None,
        expanded_args: Vec::new(),
        ice_file: None,
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
    let diagnostics = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
    println!("{diagnostics}");
}
