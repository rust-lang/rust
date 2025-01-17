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

use rustc_ast_pretty::pprust::item_to_string;
use rustc_data_structures::sync::Lrc;
use rustc_driver::{Compilation, RunCompiler};
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;

struct MyFileLoader;

impl rustc_span::source_map::FileLoader for MyFileLoader {
    fn file_exists(&self, path: &Path) -> bool {
        path == Path::new("main.rs")
    }

    fn read_file(&self, path: &Path) -> io::Result<String> {
        if path == Path::new("main.rs") {
            Ok(r#"
fn main() {
    let message = "Hello, World!";
    println!("{message}");
}
"#
            .to_string())
        } else {
            Err(io::Error::other("oops"))
        }
    }

    fn read_binary_file(&self, _path: &Path) -> io::Result<Lrc<[u8]>> {
        Err(io::Error::other("oops"))
    }
}

struct MyCallbacks;

impl rustc_driver::Callbacks for MyCallbacks {
    fn after_crate_root_parsing(
        &mut self,
        _compiler: &Compiler,
        krate: &rustc_ast::Crate,
    ) -> Compilation {
        for item in &krate.items {
            println!("{}", item_to_string(&item));
        }

        Compilation::Continue
    }

    fn after_analysis(&mut self, _compiler: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        // Every compilation contains a single crate.
        let hir_krate = tcx.hir();
        // Iterate over the top-level items in the crate, looking for the main function.
        for id in hir_krate.items() {
            let item = hir_krate.item(id);
            // Use pattern-matching to find a specific node inside the main function.
            if let rustc_hir::ItemKind::Fn(_, _, body_id) = item.kind {
                let expr = &tcx.hir().body(body_id).value;
                if let rustc_hir::ExprKind::Block(block, _) = expr.kind {
                    if let rustc_hir::StmtKind::Let(let_stmt) = block.stmts[0].kind {
                        if let Some(expr) = let_stmt.init {
                            let hir_id = expr.hir_id; // hir_id identifies the string "Hello, world!"
                            let def_id = item.hir_id().owner.def_id; // def_id identifies the main function
                            let ty = tcx.typeck(def_id).node_type(hir_id);
                            println!("{expr:#?}: {ty:?}");
                        }
                    }
                }
            }
        }

        Compilation::Stop
    }
}

fn main() {
    match RunCompiler::new(&["main.rs".to_string()], &mut MyCallbacks) {
        mut compiler => {
            compiler.set_file_loader(Some(Box::new(MyFileLoader)));
            compiler.run();
        }
    }
}
