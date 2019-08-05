use std::{collections::HashSet, fmt::Write, path::Path, time::Instant};

use ra_db::SourceDatabase;
use ra_hir::{Crate, HasSource, ImplItem, ModuleDef, Ty};
use ra_syntax::AstNode;

use crate::Result;

pub fn run(verbose: bool, memory_usage: bool, path: &Path, only: Option<&str>) -> Result<()> {
    let db_load_time = Instant::now();
    let (mut host, roots) = ra_batch::load_cargo(path)?;
    let db = host.raw_database();
    println!("Database loaded, {} roots, {:?}", roots.len(), db_load_time.elapsed());
    let analysis_time = Instant::now();
    let mut num_crates = 0;
    let mut visited_modules = HashSet::new();
    let mut visit_queue = Vec::new();
    for (source_root_id, project_root) in roots {
        if project_root.is_member() {
            for krate in Crate::source_root_crates(db, source_root_id) {
                num_crates += 1;
                let module =
                    krate.root_module(db).expect("crate in source root without root module");
                visit_queue.push(module);
            }
        }
    }
    println!("Crates in this dir: {}", num_crates);
    let mut num_decls = 0;
    let mut funcs = Vec::new();
    while let Some(module) = visit_queue.pop() {
        if visited_modules.insert(module) {
            visit_queue.extend(module.children(db));

            for decl in module.declarations(db) {
                num_decls += 1;
                if let ModuleDef::Function(f) = decl {
                    funcs.push(f);
                }
            }

            for impl_block in module.impl_blocks(db) {
                for item in impl_block.items(db) {
                    num_decls += 1;
                    if let ImplItem::Method(f) = item {
                        funcs.push(f);
                    }
                }
            }
        }
    }
    println!("Total modules found: {}", visited_modules.len());
    println!("Total declarations: {}", num_decls);
    println!("Total functions: {}", funcs.len());
    println!("Item Collection: {:?}, {}", analysis_time.elapsed(), ra_prof::memory_usage());

    let inference_time = Instant::now();
    let bar = indicatif::ProgressBar::with_draw_target(
        funcs.len() as u64,
        indicatif::ProgressDrawTarget::stderr_nohz(),
    );
    bar.set_style(
        indicatif::ProgressStyle::default_bar().template("{wide_bar} {pos}/{len}\n{msg}"),
    );
    bar.tick();
    let mut num_exprs = 0;
    let mut num_exprs_unknown = 0;
    let mut num_exprs_partially_unknown = 0;
    for f in funcs {
        let name = f.name(db);
        let mut msg = format!("processing: {}", name);
        if verbose {
            let src = f.source(db);
            let original_file = src.file_id.original_file(db);
            let path = db.file_relative_path(original_file);
            let syntax_range = src.ast.syntax().text_range();
            write!(msg, " ({:?} {})", path, syntax_range).unwrap();
        }
        bar.set_message(&msg);
        if let Some(only_name) = only {
            if name.to_string() != only_name {
                continue;
            }
        }
        let body = f.body(db);
        let inference_result = f.infer(db);
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
    println!("Inference: {:?}, {}", inference_time.elapsed(), ra_prof::memory_usage());
    println!("Total: {:?}, {}", analysis_time.elapsed(), ra_prof::memory_usage());

    if memory_usage {
        drop(db);
        for (name, bytes) in host.per_query_memory_usage() {
            println!("{:>8} {}", bytes, name)
        }
        let before = ra_prof::memory_usage();
        drop(host);
        println!("leftover: {}", before.allocated - ra_prof::memory_usage().allocated)
    }

    Ok(())
}
