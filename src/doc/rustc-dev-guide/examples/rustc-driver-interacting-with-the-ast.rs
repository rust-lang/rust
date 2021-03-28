#![feature(rustc_private)]

// NOTE: For the example to compile, you will need to first run the following:
//     rustup component add rustc-dev llvm-tools-preview

// version: 1.53.0-nightly (9b0edb7fd 2021-03-27)

extern crate rustc_ast_pretty;
extern crate rustc_error_codes;
extern crate rustc_errors;
extern crate rustc_hash;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use rustc_ast_pretty::pprust::item_to_string;
use rustc_errors::registry;
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
        opts: config::Options {
            maybe_sysroot: Some(path::PathBuf::from(sysroot)),
            ..config::Options::default()
        },
        input: config::Input::Str {
            name: source_map::FileName::Custom("main.rs".to_string()),
            input: "fn main() { let message = \"Hello, world!\"; println!(\"{}\", message); }"
                .to_string(),
        },
        diagnostic_output: rustc_session::DiagnosticOutput::Default,
        crate_cfg: rustc_hash::FxHashSet::default(),
        input_path: None,
        output_dir: None,
        output_file: None,
        file_loader: None,
        stderr: None,
        lint_caps: rustc_hash::FxHashMap::default(),
        parse_sess_created: None,
        register_lints: None,
        override_queries: None,
        make_codegen_backend: None,
        registry: registry::Registry::new(&rustc_error_codes::DIAGNOSTICS),
    };
    rustc_interface::run_compiler(config, |compiler| {
        compiler.enter(|queries| {
            // TODO: add this to -Z unpretty
            let ast_krate = queries.parse().unwrap().take();
            for item in ast_krate.items {
                println!("{}", item_to_string(&item));
            }

            // Analyze the crate and inspect the types under the cursor.
            queries.global_ctxt().unwrap().take().enter(|tcx| {
                // Every compilation contains a single crate.
                let hir_krate = tcx.hir().krate();
                // Iterate over the top-level items in the crate, looking for the main function.
                for (_, item) in &hir_krate.items {
                    // Use pattern-matching to find a specific node inside the main function.
                    if let rustc_hir::ItemKind::Fn(_, _, body_id) = item.kind {
                        let expr = &tcx.hir().body(body_id).value;
                        if let rustc_hir::ExprKind::Block(block, _) = expr.kind {
                            if let rustc_hir::StmtKind::Local(local) = block.stmts[0].kind {
                                if let Some(expr) = local.init {
                                    let hir_id = expr.hir_id; // hir_id identifies the string "Hello, world!"
                                    let def_id = tcx.hir().local_def_id(item.hir_id()); // def_id identifies the main function
                                    let ty = tcx.typeck(def_id).node_type(hir_id);
                                    println!("{:?}: {:?}", expr, ty); // prints expr(HirId { owner: DefIndex(3), local_id: 4 }: "Hello, world!"): &'static str
                                }
                            }
                        }
                    }
                }
            })
        });
    });
}
