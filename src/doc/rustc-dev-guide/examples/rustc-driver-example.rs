#![feature(rustc_private)]

extern crate rustc;
extern crate rustc_error_codes;
extern crate rustc_errors;
extern crate rustc_hash;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_span;

use rustc::session;
use rustc::session::config;
use rustc_errors::registry;
use rustc_hash::{FxHashMap, FxHashSet};
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
    let filename = "main.rs";
    let contents = "static HELLO: &str = \"Hello, world!\"; fn main() { println!(\"{}\", HELLO); }";
    let errors = registry::Registry::new(&rustc_error_codes::DIAGNOSTICS);
    let config = rustc_interface::Config {
        // Command line options
        opts: config::Options {
            maybe_sysroot: Some(path::PathBuf::from(sysroot)),
            ..config::Options::default()
        },

        // cfg! configuration in addition to the default ones
        // FxHashSet<(String, Option<String>)>
        crate_cfg: FxHashSet::default(),

        input: config::Input::Str {
            name: source_map::FileName::Custom(String::from(filename)),
            input: String::from(contents),
        },
        // Option<PathBuf>
        input_path: None,
        // Option<PathBuf>
        output_dir: None,
        // Option<PathBuf>
        output_file: None,
        // Option<Box<dyn FileLoader + Send + Sync>>
        file_loader: None,
        diagnostic_output: session::DiagnosticOutput::Default,

        // Set to capture stderr output during compiler execution
        // Option<Arc<Mutex<Vec<u8>>>>
        stderr: None,

        // Option<String>
        crate_name: None,
        // FxHashMap<lint::LintId, lint::Level>
        lint_caps: FxHashMap::default(),

        // This is a callback from the driver that is called when we're registering lints;
        // it is called during plugin registration when we have the LintStore in a non-shared state.
        //
        // Note that if you find a Some here you probably want to call that function in the new
        // function being registered.
        // Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>>
        register_lints: None,

        // This is a callback from the driver that is called just after we have populated
        // the list of queries.
        //
        // The second parameter is local providers and the third parameter is external providers.
        // Option<fn(&Session, &mut ty::query::Providers<'_>, &mut ty::query::Providers<'_>)>
        override_queries: None,

        // Registry of diagnostics codes.
        registry: errors,
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
                            let ty = tcx.type_of(tcx.hir().local_def_id(item.hir_id));
                            println!("{:?}:\t{:?}", name, ty)
                        }
                        _ => (),
                    }
                }
            })
        });
    });
}
