//! Fully type-check project and print various stats, like the number of type
//! errors.

use std::{path::Path, time::Instant};

use itertools::Itertools;
use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

use hir::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    original_range, AssocItem, Crate, HasSource, HirDisplay, ModuleDef,
};
use hir_def::FunctionId;
use hir_ty::{Ty, TypeWalk};
use ra_db::{
    salsa::{self, ParallelDatabase},
    SourceDatabaseExt,
};
use ra_syntax::AstNode;
use stdx::format_to;

use crate::cli::{load_cargo::load_cargo, progress_report::ProgressReport, Result, Verbosity};

/// Need to wrap Snapshot to provide `Clone` impl for `map_with`
struct Snap<DB>(DB);
impl<DB: ParallelDatabase> Clone for Snap<salsa::Snapshot<DB>> {
    fn clone(&self) -> Snap<salsa::Snapshot<DB>> {
        Snap(self.0.snapshot())
    }
}

pub fn analysis_stats(
    verbosity: Verbosity,
    memory_usage: bool,
    path: &Path,
    only: Option<&str>,
    with_deps: bool,
    randomize: bool,
    parallel: bool,
    load_output_dirs: bool,
    with_proc_macro: bool,
) -> Result<()> {
    let db_load_time = Instant::now();
    let (mut host, vfs) = load_cargo(path, load_output_dirs, with_proc_macro)?;
    let db = host.raw_database();
    println!("Database loaded {:?}", db_load_time.elapsed());
    let analysis_time = Instant::now();
    let mut num_crates = 0;
    let mut visited_modules = FxHashSet::default();
    let mut visit_queue = Vec::new();

    let mut krates = Crate::all(db);
    if randomize {
        krates.shuffle(&mut thread_rng());
    }
    for krate in krates {
        let module = krate.root_module(db).expect("crate without root module");
        let file_id = module.definition_source(db).file_id;
        let file_id = file_id.original_file(db);
        let source_root = db.file_source_root(file_id);
        let source_root = db.source_root(source_root);
        if !source_root.is_library || with_deps {
            num_crates += 1;
            visit_queue.push(module);
        }
    }

    if randomize {
        visit_queue.shuffle(&mut thread_rng());
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

            for impl_def in module.impl_defs(db) {
                for item in impl_def.items(db) {
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

    if randomize {
        funcs.shuffle(&mut thread_rng());
    }

    let mut bar = match verbosity {
        Verbosity::Quiet | Verbosity::Spammy => ProgressReport::hidden(),
        _ => ProgressReport::new(funcs.len() as u64),
    };

    if parallel {
        let inference_time = Instant::now();
        let snap = Snap(db.snapshot());
        funcs
            .par_iter()
            .map_with(snap, |snap, &f| {
                let f_id = FunctionId::from(f);
                snap.0.body(f_id.into());
                snap.0.infer(f_id.into());
            })
            .count();
        println!("Parallel Inference: {:?}, {}", inference_time.elapsed(), ra_prof::memory_usage());
    }

    let inference_time = Instant::now();
    bar.tick();
    let mut num_exprs = 0;
    let mut num_exprs_unknown = 0;
    let mut num_exprs_partially_unknown = 0;
    let mut num_type_mismatches = 0;
    for f in funcs {
        let name = f.name(db);
        let full_name = f
            .module(db)
            .path_to_root(db)
            .into_iter()
            .rev()
            .filter_map(|it| it.name(db))
            .chain(Some(f.name(db)))
            .join("::");
        if let Some(only_name) = only {
            if name.to_string() != only_name && full_name != only_name {
                continue;
            }
        }
        let mut msg = format!("processing: {}", full_name);
        if verbosity.is_verbose() {
            let src = f.source(db);
            let original_file = src.file_id.original_file(db);
            let path = vfs.file_path(original_file);
            let syntax_range = src.value.syntax().text_range();
            format_to!(msg, " ({} {:?})", path, syntax_range);
        }
        if verbosity.is_spammy() {
            bar.println(msg.to_string());
        }
        bar.set_message(&msg);
        let f_id = FunctionId::from(f);
        let body = db.body(f_id.into());
        let inference_result = db.infer(f_id.into());
        let (previous_exprs, previous_unknown, previous_partially_unknown) =
            (num_exprs, num_exprs_unknown, num_exprs_partially_unknown);
        for (expr_id, _) in body.exprs.iter() {
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
            if only.is_some() && verbosity.is_spammy() {
                // in super-verbose mode for just one function, we print every single expression
                let (_, sm) = db.body_with_source_map(f_id.into());
                let src = sm.expr_syntax(expr_id);
                if let Ok(src) = src {
                    let node = {
                        let root = db.parse_or_expand(src.file_id).unwrap();
                        src.value.to_node(&root)
                    };
                    let original_file = src.file_id.original_file(db);
                    let line_index = host.analysis().file_line_index(original_file).unwrap();
                    let text_range = node.syntax().text_range();
                    let (start, end) = (
                        line_index.line_col(text_range.start()),
                        line_index.line_col(text_range.end()),
                    );
                    bar.println(format!(
                        "{}:{}-{}:{}: {}",
                        start.line + 1,
                        start.col_utf16,
                        end.line + 1,
                        end.col_utf16,
                        ty.display(db)
                    ));
                } else {
                    bar.println(format!("unknown location: {}", ty.display(db)));
                }
            }
            if let Some(mismatch) = inference_result.type_mismatch_for_expr(expr_id) {
                num_type_mismatches += 1;
                if verbosity.is_verbose() {
                    let (_, sm) = db.body_with_source_map(f_id.into());
                    let src = sm.expr_syntax(expr_id);
                    if let Ok(src) = src {
                        // FIXME: it might be nice to have a function (on Analysis?) that goes from Source<T> -> (LineCol, LineCol) directly
                        // But also, we should just turn the type mismatches into diagnostics and provide these
                        let root = db.parse_or_expand(src.file_id).unwrap();
                        let node = src.map(|e| e.to_node(&root).syntax().clone());
                        let original_range = original_range(db, node.as_ref());
                        let path = vfs.file_path(original_range.file_id);
                        let line_index =
                            host.analysis().file_line_index(original_range.file_id).unwrap();
                        let text_range = original_range.range;
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
        if verbosity.is_spammy() {
            bar.println(format!(
                "In {}: {} exprs, {} unknown, {} partial",
                full_name,
                num_exprs - previous_exprs,
                num_exprs_unknown - previous_unknown,
                num_exprs_partially_unknown - previous_partially_unknown
            ));
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
    println!("Total expressions: {}", num_exprs);
    println!(
        "Expressions of unknown type: {} ({}%)",
        num_exprs_unknown,
        if num_exprs > 0 { num_exprs_unknown * 100 / num_exprs } else { 100 }
    );
    println!(
        "Expressions of partially unknown type: {} ({}%)",
        num_exprs_partially_unknown,
        if num_exprs > 0 { num_exprs_partially_unknown * 100 / num_exprs } else { 100 }
    );
    println!("Type mismatches: {}", num_type_mismatches);
    println!("Inference: {:?}, {}", inference_time.elapsed(), ra_prof::memory_usage());
    println!("Total: {:?}, {}", analysis_time.elapsed(), ra_prof::memory_usage());

    if memory_usage {
        let mut mem = host.per_query_memory_usage();

        let before = ra_prof::memory_usage();
        drop(vfs);
        let vfs = before.allocated - ra_prof::memory_usage().allocated;
        mem.push(("VFS".into(), vfs));

        let before = ra_prof::memory_usage();
        drop(host);
        mem.push(("Unaccounted".into(), before.allocated - ra_prof::memory_usage().allocated));

        mem.push(("Remaining".into(), ra_prof::memory_usage().allocated));

        for (name, bytes) in mem {
            println!("{:>8} {}", bytes, name)
        }
    }

    Ok(())
}
