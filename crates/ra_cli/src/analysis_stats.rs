//! FIXME: write short doc here

use std::{collections::HashSet, fmt::Write, path::Path, time::Instant};

use ra_db::SourceDatabaseExt;
use ra_hir::{AssocItem, Crate, HasBodySource, HasSource, HirDisplay, ModuleDef, Ty, TypeWalk};
use ra_syntax::AstNode;

use crate::{Result, Verbosity};

pub fn run(
    verbosity: Verbosity,
    memory_usage: bool,
    path: &Path,
    only: Option<&str>,
    with_deps: bool,
) -> Result<()> {
    let db_load_time = Instant::now();
    let (mut host, roots) = ra_batch::load_cargo(path)?;
    let db = host.raw_database();
    println!("Database loaded, {} roots, {:?}", roots.len(), db_load_time.elapsed());
    let analysis_time = Instant::now();
    let mut num_crates = 0;
    let mut visited_modules = HashSet::new();
    let mut visit_queue = Vec::new();

    let members =
        roots
            .into_iter()
            .filter_map(|(source_root_id, project_root)| {
                if with_deps || project_root.is_member() {
                    Some(source_root_id)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();

    for krate in Crate::all(db) {
        let module = krate.root_module(db).expect("crate without root module");
        let file_id = module.definition_source(db).file_id;
        if members.contains(&db.file_source_root(file_id.original_file(db))) {
            num_crates += 1;
            visit_queue.push(module);
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
                    if let AssocItem::Function(f) = item {
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
    let bar = match verbosity {
        Verbosity::Verbose | Verbosity::Normal => indicatif::ProgressBar::with_draw_target(
            funcs.len() as u64,
            indicatif::ProgressDrawTarget::stderr_nohz(),
        ),
        Verbosity::Quiet => indicatif::ProgressBar::hidden(),
    };

    bar.set_style(
        indicatif::ProgressStyle::default_bar().template("{wide_bar} {pos}/{len}\n{msg}"),
    );
    bar.tick();
    let mut num_exprs = 0;
    let mut num_exprs_unknown = 0;
    let mut num_exprs_partially_unknown = 0;
    let mut num_type_mismatches = 0;
    for f in funcs {
        let name = f.name(db);
        let mut msg = format!("processing: {}", name);
        if verbosity.is_verbose() {
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
            if let Some(mismatch) = inference_result.type_mismatch_for_expr(expr_id) {
                num_type_mismatches += 1;
                if verbosity.is_verbose() {
                    let src = f.expr_source(db, expr_id);
                    if let Some(src) = src {
                        // FIXME: it might be nice to have a function (on Analysis?) that goes from Source<T> -> (LineCol, LineCol) directly
                        let original_file = src.file_id.original_file(db);
                        let path = db.file_relative_path(original_file);
                        let line_index = host.analysis().file_line_index(original_file).unwrap();
                        let text_range = src
                            .ast
                            .either(|it| it.syntax().text_range(), |it| it.syntax().text_range());
                        let (start, end) = (
                            line_index.line_col(text_range.start()),
                            line_index.line_col(text_range.end()),
                        );
                        bar.println(format!(
                            "{} {}:{}-{}:{}: Expected {}, got {}",
                            path,
                            start.line + 1,
                            start.col_utf16,
                            end.line + 1,
                            end.col_utf16,
                            mismatch.expected.display(db),
                            mismatch.actual.display(db)
                        ));
                    } else {
                        bar.println(format!(
                            "{}: Expected {}, got {}",
                            name,
                            mismatch.expected.display(db),
                            mismatch.actual.display(db)
                        ));
                    }
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
        if num_exprs > 0 { (num_exprs_unknown * 100 / num_exprs) } else { 100 }
    );
    println!(
        "Expressions of partially unknown type: {} ({}%)",
        num_exprs_partially_unknown,
        if num_exprs > 0 { (num_exprs_partially_unknown * 100 / num_exprs) } else { 100 }
    );
    println!("Type mismatches: {}", num_type_mismatches);
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
