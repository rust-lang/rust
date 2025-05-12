// Tested with nightly-2025-03-28

#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_error_codes;
extern crate rustc_errors;
extern crate rustc_hash;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

use std::io;
use std::path::Path;
use std::sync::Arc;

use rustc_ast_pretty::pprust::item_to_string;
use rustc_driver::{Compilation, run_compiler};
use rustc_interface::interface::{Compiler, Config};
use rustc_middle::ty::TyCtxt;

struct MyFileLoader;

impl rustc_span::source_map::FileLoader for MyFileLoader {
    fn file_exists(&self, path: &Path) -> bool {
        path == Path::new("main.rs")
    }

    fn read_file(&self, path: &Path) -> io::Result<String> {
        if path == Path::new("main.rs") {
            Ok(r#"
static MESSAGE: &str = "Hello, World!";
fn main() {
    println!("{MESSAGE}");
}
"#
            .to_string())
        } else {
            Err(io::Error::other("oops"))
        }
    }

    fn read_binary_file(&self, _path: &Path) -> io::Result<Arc<[u8]>> {
        Err(io::Error::other("oops"))
    }
}

struct MyCallbacks;

impl rustc_driver::Callbacks for MyCallbacks {
    fn config(&mut self, config: &mut Config) {
        config.file_loader = Some(Box::new(MyFileLoader));
    }

    fn after_crate_root_parsing(
        &mut self,
        _compiler: &Compiler,
        krate: &mut rustc_ast::Crate,
    ) -> Compilation {
        for item in &krate.items {
            println!("{}", item_to_string(&item));
        }

        Compilation::Continue
    }

    fn after_analysis(&mut self, _compiler: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        // Analyze the program and inspect the types of definitions.
        for id in tcx.hir_free_items() {
            let item = &tcx.hir_item(id);
            match item.kind {
                rustc_hir::ItemKind::Static(ident, ..) | rustc_hir::ItemKind::Fn { ident, .. } => {
                    let ty = tcx.type_of(item.hir_id().owner.def_id);
                    println!("{ident:?}:\t{ty:?}")
                }
                _ => (),
            }
        }

        Compilation::Stop
    }
}

fn main() {
    run_compiler(
        &[
            // The first argument, which in practice contains the name of the binary being executed
            // (i.e. "rustc") is ignored by rustc.
            "ignored".to_string(),
            "main.rs".to_string(),
        ],
        &mut MyCallbacks,
    );
}
