#![feature(rustc_private)]

// NOTE: For the example to compile, you will need to first run the following:
//   rustup component add rustc-dev

// version: 1.53.0-nightly (9b0edb7fd 2021-03-27)

extern crate rustc_error_codes;
extern crate rustc_errors;
extern crate rustc_hash;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use rustc_errors::registry;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_session::config;
use rustc_span::source_map;
use std::path;
use std::process;
use std::str;

fn main() {
    let out = process::Command::new("rustc")
        .arg("--print=sysroot")
        .current_dir(".")
        .output()
        .unwrap();
    let sysroot = str::from_utf8(&out.stdout).unwrap().trim();
    let config = rustc_interface::Config {
        // Command line options
        opts: config::Options {
            maybe_sysroot: Some(path::PathBuf::from(sysroot)),
            ..config::Options::default()
        },
        // cfg! configuration in addition to the default ones
        crate_cfg: FxHashSet::default(), // FxHashSet<(String, Option<String>)>
        input: config::Input::Str {
            name: source_map::FileName::Custom("main.rs".to_string()),
            input: "static HELLO: &str = \"Hello, world!\"; fn main() { println!(\"{}\", HELLO); }"
                .to_string(),
        },
        input_path: None,  // Option<PathBuf>
        output_dir: None,  // Option<PathBuf>
        output_file: None, // Option<PathBuf>
        file_loader: None, // Option<Box<dyn FileLoader + Send + Sync>>
        diagnostic_output: rustc_session::DiagnosticOutput::Default,
        // Set to capture stderr output during compiler execution
        stderr: None,                    // Option<Arc<Mutex<Vec<u8>>>>
        lint_caps: FxHashMap::default(), // FxHashMap<lint::LintId, lint::Level>
        // This is a callback from the driver that is called when [`ParseSess`] is created.
        parse_sess_created: None, //Option<Box<dyn FnOnce(&mut ParseSess) + Send>>
        // This is a callback from the driver that is called when we're registering lints;
        // it is called during plugin registration when we have the LintStore in a non-shared state.
        //
        // Note that if you find a Some here you probably want to call that function in the new
        // function being registered.
        register_lints: None, // Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>>
        // This is a callback from the driver that is called just after we have populated
        // the list of queries.
        //
        // The second parameter is local providers and the third parameter is external providers.
        override_queries: None, // Option<fn(&Session, &mut ty::query::Providers<'_>, &mut ty::query::Providers<'_>)>
        // Registry of diagnostics codes.
        registry: registry::Registry::new(&rustc_error_codes::DIAGNOSTICS),
        make_codegen_backend: None,
    };
    rustc_interface::run_compiler(config, |compiler| {
        compiler.enter(|queries| {
            // Parse the program and print the syntax tree.
            let parse = queries.parse().unwrap().take();
            println!("{:#?}", parse);
            // Analyze the program and inspect the types of definitions.
            queries.global_ctxt().unwrap().take().enter(|tcx| {
                for (_, item) in &tcx.hir().krate().items {
                    match item.kind {
                        rustc_hir::ItemKind::Static(_, _, _) | rustc_hir::ItemKind::Fn(_, _, _) => {
                            let name = item.ident;
                            let ty = tcx.type_of(tcx.hir().local_def_id(item.hir_id()));
                            println!("{:?}:\t{:?}", name, ty)
                        }
                        _ => (),
                    }
                }
            })
        });
    });
}
