use std::collections::HashSet;

use ra_db::SourceDatabase;
use ra_batch::BatchDatabase;
use ra_hir::{Crate, ModuleDef, Ty, ImplItem};
use ra_syntax::AstNode;

use crate::Result;

pub fn run(verbose: bool) -> Result<()> {
    let (db, roots) = BatchDatabase::load_cargo(".")?;
    println!("Database loaded, {} roots", roots.len());
    let mut num_crates = 0;
    let mut visited_modules = HashSet::new();
    let mut visit_queue = Vec::new();
    for root in roots {
        for krate in Crate::source_root_crates(&db, root) {
            num_crates += 1;
            let module = krate.root_module(&db).expect("crate in source root without root module");
            visit_queue.push(module);
        }
    }
    println!("Crates in this dir: {}", num_crates);
    let mut num_decls = 0;
    let mut funcs = Vec::new();
    while let Some(module) = visit_queue.pop() {
        if visited_modules.insert(module) {
            visit_queue.extend(module.children(&db));

            for decl in module.declarations(&db) {
                num_decls += 1;
                match decl {
                    ModuleDef::Function(f) => funcs.push(f),
                    _ => {}
                }
            }

            for impl_block in module.impl_blocks(&db) {
                for item in impl_block.items() {
                    num_decls += 1;
                    match item {
                        ImplItem::Method(f) => funcs.push(*f),
                        _ => {}
                    }
                }
            }
        }
    }
    println!("Total modules found: {}", visited_modules.len());
    println!("Total declarations: {}", num_decls);
    println!("Total functions: {}", funcs.len());
    let bar = indicatif::ProgressBar::new(funcs.len() as u64);
    bar.tick();
    let mut num_exprs = 0;
    let mut num_exprs_unknown = 0;
    let mut num_exprs_partially_unknown = 0;
    for f in funcs {
        if verbose {
            let (file_id, source) = f.source(&db);
            let original_file = file_id.original_file(&db);
            let path = db.file_relative_path(original_file);
            let syntax_range = source.syntax().range();
            let name = f.name(&db);
            println!("{} ({:?} {})", name, path, syntax_range);
        }
        let body = f.body(&db);
        let inference_result = f.infer(&db);
        for (expr_id, _) in body.exprs() {
            let ty = &inference_result[expr_id];
            num_exprs += 1;
            if let Ty::Unknown = ty {
                num_exprs_unknown += 1;
            } else {
                let mut is_partially_unknown = false;
                ty.walk(&mut |ty| {
                    if let Ty::Unknown = ty {
                        is_partially_unknown = true;
                    }
                });
                if is_partially_unknown {
                    num_exprs_partially_unknown += 1;
                }
            }
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
    println!("Total expressions: {}", num_exprs);
    println!(
        "Expressions of unknown type: {} ({}%)",
        num_exprs_unknown,
        (num_exprs_unknown * 100 / num_exprs)
    );
    println!(
        "Expressions of partially unknown type: {} ({}%)",
        num_exprs_partially_unknown,
        (num_exprs_partially_unknown * 100 / num_exprs)
    );
    Ok(())
}
