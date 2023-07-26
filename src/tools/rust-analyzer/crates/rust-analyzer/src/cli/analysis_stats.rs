//! Fully type-check project and print various stats, like the number of type
//! errors.

use std::{
    env,
    time::{SystemTime, UNIX_EPOCH},
};

use hir::{
    db::{DefDatabase, ExpandDatabase, HirDatabase},
    Adt, AssocItem, Crate, DefWithBody, HasSource, HirDisplay, ModuleDef, Name,
};
use hir_def::{
    body::{BodySourceMap, SyntheticSyntax},
    hir::{ExprId, PatId},
};
use hir_ty::{Interner, Substitution, TyExt, TypeFlags};
use ide::{Analysis, AnnotationConfig, DiagnosticsConfig, InlayHintsConfig, LineCol, RootDatabase};
use ide_db::{
    base_db::{
        salsa::{self, debug::DebugQueryTable, ParallelDatabase},
        SourceDatabase, SourceDatabaseExt,
    },
    LineIndexDatabase,
};
use itertools::Itertools;
use load_cargo::{load_workspace, LoadCargoConfig, ProcMacroServerChoice};
use oorandom::Rand32;
use profile::{Bytes, StopWatch};
use project_model::{CargoConfig, ProjectManifest, ProjectWorkspace, RustLibSource};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use syntax::{AstNode, SyntaxNode};
use vfs::{AbsPathBuf, FileId, Vfs, VfsPath};

use crate::cli::{
    flags::{self, OutputFormat},
    full_name_of_item, print_memory_usage,
    progress_report::ProgressReport,
    report_metric, Verbosity,
};

/// Need to wrap Snapshot to provide `Clone` impl for `map_with`
struct Snap<DB>(DB);
impl<DB: ParallelDatabase> Clone for Snap<salsa::Snapshot<DB>> {
    fn clone(&self) -> Snap<salsa::Snapshot<DB>> {
        Snap(self.0.snapshot())
    }
}

impl flags::AnalysisStats {
    pub fn run(self, verbosity: Verbosity) -> anyhow::Result<()> {
        let mut rng = {
            let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
            Rand32::new(seed)
        };

        let mut cargo_config = CargoConfig::default();
        cargo_config.sysroot = match self.no_sysroot {
            true => None,
            false => Some(RustLibSource::Discover),
        };
        let no_progress = &|_| ();

        let mut db_load_sw = self.stop_watch();

        let path = AbsPathBuf::assert(env::current_dir()?.join(&self.path));
        let manifest = ProjectManifest::discover_single(&path)?;

        let mut workspace = ProjectWorkspace::load(manifest, &cargo_config, no_progress)?;
        let metadata_time = db_load_sw.elapsed();
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: !self.disable_build_scripts,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: false,
        };

        let build_scripts_time = if self.disable_build_scripts {
            None
        } else {
            let mut build_scripts_sw = self.stop_watch();
            let bs = workspace.run_build_scripts(&cargo_config, no_progress)?;
            workspace.set_build_scripts(bs);
            Some(build_scripts_sw.elapsed())
        };

        let (host, vfs, _proc_macro) =
            load_workspace(workspace, &cargo_config.extra_env, &load_cargo_config)?;
        let db = host.raw_database();
        eprint!("{:<20} {}", "Database loaded:", db_load_sw.elapsed());
        eprint!(" (metadata {metadata_time}");
        if let Some(build_scripts_time) = build_scripts_time {
            eprint!("; build {build_scripts_time}");
        }
        eprintln!(")");

        let mut analysis_sw = self.stop_watch();

        let mut krates = Crate::all(db);
        if self.randomize {
            shuffle(&mut rng, &mut krates);
        }

        let mut item_tree_sw = self.stop_watch();
        let mut num_item_trees = 0;
        let source_roots =
            krates.iter().cloned().map(|krate| db.file_source_root(krate.root_file(db))).unique();
        for source_root_id in source_roots {
            let source_root = db.source_root(source_root_id);
            if !source_root.is_library || self.with_deps {
                for file_id in source_root.iter() {
                    if let Some(p) = source_root.path_for_file(&file_id) {
                        if let Some((_, Some("rs"))) = p.name_and_extension() {
                            db.file_item_tree(file_id.into());
                            num_item_trees += 1;
                        }
                    }
                }
            }
        }
        eprintln!("  item trees: {num_item_trees}");
        let item_tree_time = item_tree_sw.elapsed();
        eprintln!("{:<20} {}", "Item Tree Collection:", item_tree_time);
        report_metric("item tree time", item_tree_time.time.as_millis() as u64, "ms");

        let mut crate_def_map_sw = self.stop_watch();
        let mut num_crates = 0;
        let mut visited_modules = FxHashSet::default();
        let mut visit_queue = Vec::new();
        for krate in krates {
            let module = krate.root_module(db);
            let file_id = module.definition_source_file_id(db);
            let file_id = file_id.original_file(db);
            let source_root = db.file_source_root(file_id);
            let source_root = db.source_root(source_root);
            if !source_root.is_library || self.with_deps {
                num_crates += 1;
                visit_queue.push(module);
            }
        }

        if self.randomize {
            shuffle(&mut rng, &mut visit_queue);
        }

        eprint!("  crates: {num_crates}");
        let mut num_decls = 0;
        let mut bodies = Vec::new();
        let mut adts = Vec::new();
        let mut consts = Vec::new();
        let mut file_ids = Vec::new();
        while let Some(module) = visit_queue.pop() {
            if visited_modules.insert(module) {
                file_ids.extend(module.as_source_file_id(db));
                visit_queue.extend(module.children(db));

                for decl in module.declarations(db) {
                    num_decls += 1;
                    match decl {
                        ModuleDef::Function(f) => bodies.push(DefWithBody::from(f)),
                        ModuleDef::Adt(a) => {
                            if let Adt::Enum(e) = a {
                                for v in e.variants(db) {
                                    bodies.push(DefWithBody::from(v));
                                }
                            }
                            adts.push(a)
                        }
                        ModuleDef::Const(c) => {
                            bodies.push(DefWithBody::from(c));
                            consts.push(c)
                        }
                        ModuleDef::Static(s) => bodies.push(DefWithBody::from(s)),
                        _ => (),
                    };
                }

                for impl_def in module.impl_defs(db) {
                    for item in impl_def.items(db) {
                        num_decls += 1;
                        match item {
                            AssocItem::Function(f) => bodies.push(DefWithBody::from(f)),
                            AssocItem::Const(c) => {
                                bodies.push(DefWithBody::from(c));
                                consts.push(c);
                            }
                            _ => (),
                        }
                    }
                }
            }
        }
        eprintln!(
            ", mods: {}, decls: {num_decls}, bodies: {}, adts: {}, consts: {}",
            visited_modules.len(),
            bodies.len(),
            adts.len(),
            consts.len(),
        );
        let crate_def_map_time = crate_def_map_sw.elapsed();
        eprintln!("{:<20} {}", "Item Collection:", crate_def_map_time);
        report_metric("crate def map time", crate_def_map_time.time.as_millis() as u64, "ms");

        if self.randomize {
            shuffle(&mut rng, &mut bodies);
        }

        if !self.skip_lowering {
            self.run_body_lowering(db, &vfs, &bodies, verbosity);
        }

        if !self.skip_inference {
            self.run_inference(db, &vfs, &bodies, verbosity);
        }

        if !self.skip_mir_stats {
            self.run_mir_lowering(db, &bodies, verbosity);
        }

        if !self.skip_data_layout {
            self.run_data_layout(db, &adts, verbosity);
        }

        if !self.skip_const_eval {
            self.run_const_eval(db, &consts, verbosity);
        }

        if self.run_all_ide_things {
            self.run_ide_things(host.analysis(), file_ids);
        }

        let total_span = analysis_sw.elapsed();
        eprintln!("{:<20} {total_span}", "Total:");
        report_metric("total time", total_span.time.as_millis() as u64, "ms");
        if let Some(instructions) = total_span.instructions {
            report_metric("total instructions", instructions, "#instr");
        }
        if let Some(memory) = total_span.memory {
            report_metric("total memory", memory.allocated.megabytes() as u64, "MB");
        }

        if env::var("RA_COUNT").is_ok() {
            eprintln!("{}", profile::countme::get_all());
        }

        if self.source_stats {
            let mut total_file_size = Bytes::default();
            for e in ide_db::base_db::ParseQuery.in_db(db).entries::<Vec<_>>() {
                total_file_size += syntax_len(db.parse(e.key).syntax_node())
            }

            let mut total_macro_file_size = Bytes::default();
            for e in hir::db::ParseMacroExpansionQuery.in_db(db).entries::<Vec<_>>() {
                let val = db.parse_macro_expansion(e.key).value.0;
                total_macro_file_size += syntax_len(val.syntax_node())
            }
            eprintln!("source files: {total_file_size}, macro files: {total_macro_file_size}");
        }

        if self.memory_usage && verbosity.is_verbose() {
            print_memory_usage(host, vfs);
        }

        Ok(())
    }

    fn run_data_layout(&self, db: &RootDatabase, adts: &[hir::Adt], verbosity: Verbosity) {
        let mut sw = self.stop_watch();
        let mut all = 0;
        let mut fail = 0;
        for &a in adts {
            if db.generic_params(a.into()).iter().next().is_some() {
                // Data types with generics don't have layout.
                continue;
            }
            all += 1;
            let Err(e) = db.layout_of_adt(
                hir_def::AdtId::from(a).into(),
                Substitution::empty(Interner),
                db.trait_environment(a.into()),
            ) else {
                continue;
            };
            if verbosity.is_spammy() {
                let full_name = full_name_of_item(db, a.module(db), a.name(db));
                println!("Data layout for {full_name} failed due {e:?}");
            }
            fail += 1;
        }
        let data_layout_time = sw.elapsed();
        eprintln!("{:<20} {}", "Data layouts:", data_layout_time);
        eprintln!("Failed data layouts: {fail} ({}%)", percentage(fail, all));
        report_metric("failed data layouts", fail, "#");
        report_metric("data layout time", data_layout_time.time.as_millis() as u64, "ms");
    }

    fn run_const_eval(&self, db: &RootDatabase, consts: &[hir::Const], verbosity: Verbosity) {
        let mut sw = self.stop_watch();
        let mut all = 0;
        let mut fail = 0;
        for &c in consts {
            all += 1;
            let Err(e) = c.render_eval(db) else {
                continue;
            };
            if verbosity.is_spammy() {
                let full_name =
                    full_name_of_item(db, c.module(db), c.name(db).unwrap_or(Name::missing()));
                println!("Const eval for {full_name} failed due {e:?}");
            }
            fail += 1;
        }
        let const_eval_time = sw.elapsed();
        eprintln!("{:<20} {}", "Const evaluation:", const_eval_time);
        eprintln!("Failed const evals: {fail} ({}%)", percentage(fail, all));
        report_metric("failed const evals", fail, "#");
        report_metric("const eval time", const_eval_time.time.as_millis() as u64, "ms");
    }

    fn run_mir_lowering(&self, db: &RootDatabase, bodies: &[DefWithBody], verbosity: Verbosity) {
        let mut sw = self.stop_watch();
        let all = bodies.len() as u64;
        let mut fail = 0;
        for &body in bodies {
            let Err(e) = db.mir_body(body.into()) else {
                continue;
            };
            if verbosity.is_spammy() {
                let full_name = body
                    .module(db)
                    .path_to_root(db)
                    .into_iter()
                    .rev()
                    .filter_map(|it| it.name(db))
                    .chain(Some(body.name(db).unwrap_or_else(Name::missing)))
                    .map(|it| it.display(db).to_string())
                    .join("::");
                println!("Mir body for {full_name} failed due {e:?}");
            }
            fail += 1;
        }
        let mir_lowering_time = sw.elapsed();
        eprintln!("{:<20} {}", "MIR lowering:", mir_lowering_time);
        eprintln!("Mir failed bodies: {fail} ({}%)", percentage(fail, all));
        report_metric("mir failed bodies", fail, "#");
        report_metric("mir lowering time", mir_lowering_time.time.as_millis() as u64, "ms");
    }

    fn run_inference(
        &self,
        db: &RootDatabase,
        vfs: &Vfs,
        bodies: &[DefWithBody],
        verbosity: Verbosity,
    ) {
        let mut bar = match verbosity {
            Verbosity::Quiet | Verbosity::Spammy => ProgressReport::hidden(),
            _ if self.parallel || self.output.is_some() => ProgressReport::hidden(),
            _ => ProgressReport::new(bodies.len() as u64),
        };

        if self.parallel {
            let mut inference_sw = self.stop_watch();
            let snap = Snap(db.snapshot());
            bodies
                .par_iter()
                .map_with(snap, |snap, &body| {
                    snap.0.body(body.into());
                    snap.0.infer(body.into());
                })
                .count();
            eprintln!("{:<20} {}", "Parallel Inference:", inference_sw.elapsed());
        }

        let mut inference_sw = self.stop_watch();
        bar.tick();
        let mut num_exprs = 0;
        let mut num_exprs_unknown = 0;
        let mut num_exprs_partially_unknown = 0;
        let mut num_expr_type_mismatches = 0;
        let mut num_pats = 0;
        let mut num_pats_unknown = 0;
        let mut num_pats_partially_unknown = 0;
        let mut num_pat_type_mismatches = 0;
        for &body_id in bodies {
            let name = body_id.name(db).unwrap_or_else(Name::missing);
            let module = body_id.module(db);
            let full_name = move || {
                module
                    .krate()
                    .display_name(db)
                    .map(|it| it.canonical_name().to_string())
                    .into_iter()
                    .chain(
                        module
                            .path_to_root(db)
                            .into_iter()
                            .filter_map(|it| it.name(db))
                            .rev()
                            .chain(Some(body_id.name(db).unwrap_or_else(Name::missing)))
                            .map(|it| it.display(db).to_string()),
                    )
                    .join("::")
            };
            if let Some(only_name) = self.only.as_deref() {
                if name.display(db).to_string() != only_name && full_name() != only_name {
                    continue;
                }
            }
            let msg = move || {
                if verbosity.is_verbose() {
                    let source = match body_id {
                        DefWithBody::Function(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Static(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Const(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Variant(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::InTypeConst(_) => unimplemented!(),
                    };
                    if let Some(src) = source {
                        let original_file = src.file_id.original_file(db);
                        let path = vfs.file_path(original_file);
                        let syntax_range = src.value.text_range();
                        format!("processing: {} ({} {:?})", full_name(), path, syntax_range)
                    } else {
                        format!("processing: {}", full_name())
                    }
                } else {
                    format!("processing: {}", full_name())
                }
            };
            if verbosity.is_spammy() {
                bar.println(msg());
            }
            bar.set_message(msg);
            let (body, sm) = db.body_with_source_map(body_id.into());
            let inference_result = db.infer(body_id.into());

            // region:expressions
            let (previous_exprs, previous_unknown, previous_partially_unknown) =
                (num_exprs, num_exprs_unknown, num_exprs_partially_unknown);
            for (expr_id, _) in body.exprs.iter() {
                let ty = &inference_result[expr_id];
                num_exprs += 1;
                let unknown_or_partial = if ty.is_unknown() {
                    num_exprs_unknown += 1;
                    if verbosity.is_spammy() {
                        if let Some((path, start, end)) = expr_syntax_range(db, vfs, &sm, expr_id) {
                            bar.println(format!(
                                "{} {}:{}-{}:{}: Unknown type",
                                path,
                                start.line + 1,
                                start.col,
                                end.line + 1,
                                end.col,
                            ));
                        } else {
                            bar.println(format!("{}: Unknown type", name.display(db)));
                        }
                    }
                    true
                } else {
                    let is_partially_unknown =
                        ty.data(Interner).flags.contains(TypeFlags::HAS_ERROR);
                    if is_partially_unknown {
                        num_exprs_partially_unknown += 1;
                    }
                    is_partially_unknown
                };
                if self.only.is_some() && verbosity.is_spammy() {
                    // in super-verbose mode for just one function, we print every single expression
                    if let Some((_, start, end)) = expr_syntax_range(db, vfs, &sm, expr_id) {
                        bar.println(format!(
                            "{}:{}-{}:{}: {}",
                            start.line + 1,
                            start.col,
                            end.line + 1,
                            end.col,
                            ty.display(db)
                        ));
                    } else {
                        bar.println(format!("unknown location: {}", ty.display(db)));
                    }
                }
                if unknown_or_partial && self.output == Some(OutputFormat::Csv) {
                    println!(
                        r#"{},type,"{}""#,
                        location_csv_expr(db, vfs, &sm, expr_id),
                        ty.display(db)
                    );
                }
                if let Some(mismatch) = inference_result.type_mismatch_for_expr(expr_id) {
                    num_expr_type_mismatches += 1;
                    if verbosity.is_verbose() {
                        if let Some((path, start, end)) = expr_syntax_range(db, vfs, &sm, expr_id) {
                            bar.println(format!(
                                "{} {}:{}-{}:{}: Expected {}, got {}",
                                path,
                                start.line + 1,
                                start.col,
                                end.line + 1,
                                end.col,
                                mismatch.expected.display(db),
                                mismatch.actual.display(db)
                            ));
                        } else {
                            bar.println(format!(
                                "{}: Expected {}, got {}",
                                name.display(db),
                                mismatch.expected.display(db),
                                mismatch.actual.display(db)
                            ));
                        }
                    }
                    if self.output == Some(OutputFormat::Csv) {
                        println!(
                            r#"{},mismatch,"{}","{}""#,
                            location_csv_expr(db, vfs, &sm, expr_id),
                            mismatch.expected.display(db),
                            mismatch.actual.display(db)
                        );
                    }
                }
            }
            if verbosity.is_spammy() {
                bar.println(format!(
                    "In {}: {} exprs, {} unknown, {} partial",
                    full_name(),
                    num_exprs - previous_exprs,
                    num_exprs_unknown - previous_unknown,
                    num_exprs_partially_unknown - previous_partially_unknown
                ));
            }
            // endregion:expressions

            // region:patterns
            let (previous_pats, previous_unknown, previous_partially_unknown) =
                (num_pats, num_pats_unknown, num_pats_partially_unknown);
            for (pat_id, _) in body.pats.iter() {
                let ty = &inference_result[pat_id];
                num_pats += 1;
                let unknown_or_partial = if ty.is_unknown() {
                    num_pats_unknown += 1;
                    if verbosity.is_spammy() {
                        if let Some((path, start, end)) = pat_syntax_range(db, vfs, &sm, pat_id) {
                            bar.println(format!(
                                "{} {}:{}-{}:{}: Unknown type",
                                path,
                                start.line + 1,
                                start.col,
                                end.line + 1,
                                end.col,
                            ));
                        } else {
                            bar.println(format!("{}: Unknown type", name.display(db)));
                        }
                    }
                    true
                } else {
                    let is_partially_unknown =
                        ty.data(Interner).flags.contains(TypeFlags::HAS_ERROR);
                    if is_partially_unknown {
                        num_pats_partially_unknown += 1;
                    }
                    is_partially_unknown
                };
                if self.only.is_some() && verbosity.is_spammy() {
                    // in super-verbose mode for just one function, we print every single pattern
                    if let Some((_, start, end)) = pat_syntax_range(db, vfs, &sm, pat_id) {
                        bar.println(format!(
                            "{}:{}-{}:{}: {}",
                            start.line + 1,
                            start.col,
                            end.line + 1,
                            end.col,
                            ty.display(db)
                        ));
                    } else {
                        bar.println(format!("unknown location: {}", ty.display(db)));
                    }
                }
                if unknown_or_partial && self.output == Some(OutputFormat::Csv) {
                    println!(
                        r#"{},type,"{}""#,
                        location_csv_pat(db, vfs, &sm, pat_id),
                        ty.display(db)
                    );
                }
                if let Some(mismatch) = inference_result.type_mismatch_for_pat(pat_id) {
                    num_pat_type_mismatches += 1;
                    if verbosity.is_verbose() {
                        if let Some((path, start, end)) = pat_syntax_range(db, vfs, &sm, pat_id) {
                            bar.println(format!(
                                "{} {}:{}-{}:{}: Expected {}, got {}",
                                path,
                                start.line + 1,
                                start.col,
                                end.line + 1,
                                end.col,
                                mismatch.expected.display(db),
                                mismatch.actual.display(db)
                            ));
                        } else {
                            bar.println(format!(
                                "{}: Expected {}, got {}",
                                name.display(db),
                                mismatch.expected.display(db),
                                mismatch.actual.display(db)
                            ));
                        }
                    }
                    if self.output == Some(OutputFormat::Csv) {
                        println!(
                            r#"{},mismatch,"{}","{}""#,
                            location_csv_pat(db, vfs, &sm, pat_id),
                            mismatch.expected.display(db),
                            mismatch.actual.display(db)
                        );
                    }
                }
            }
            if verbosity.is_spammy() {
                bar.println(format!(
                    "In {}: {} pats, {} unknown, {} partial",
                    full_name(),
                    num_pats - previous_pats,
                    num_pats_unknown - previous_unknown,
                    num_pats_partially_unknown - previous_partially_unknown
                ));
            }
            // endregion:patterns
            bar.inc(1);
        }

        bar.finish_and_clear();
        let inference_time = inference_sw.elapsed();
        eprintln!(
            "  exprs: {}, ??ty: {} ({}%), ?ty: {} ({}%), !ty: {}",
            num_exprs,
            num_exprs_unknown,
            percentage(num_exprs_unknown, num_exprs),
            num_exprs_partially_unknown,
            percentage(num_exprs_partially_unknown, num_exprs),
            num_expr_type_mismatches
        );
        eprintln!(
            "  pats: {}, ??ty: {} ({}%), ?ty: {} ({}%), !ty: {}",
            num_pats,
            num_pats_unknown,
            percentage(num_pats_unknown, num_pats),
            num_pats_partially_unknown,
            percentage(num_pats_partially_unknown, num_pats),
            num_pat_type_mismatches
        );
        eprintln!("{:<20} {}", "Inference:", inference_time);
        report_metric("unknown type", num_exprs_unknown, "#");
        report_metric("type mismatches", num_expr_type_mismatches, "#");
        report_metric("pattern unknown type", num_pats_unknown, "#");
        report_metric("pattern type mismatches", num_pat_type_mismatches, "#");
        report_metric("inference time", inference_time.time.as_millis() as u64, "ms");
    }

    fn run_body_lowering(
        &self,
        db: &RootDatabase,
        vfs: &Vfs,
        bodies: &[DefWithBody],
        verbosity: Verbosity,
    ) {
        let mut bar = match verbosity {
            Verbosity::Quiet | Verbosity::Spammy => ProgressReport::hidden(),
            _ if self.output.is_some() => ProgressReport::hidden(),
            _ => ProgressReport::new(bodies.len() as u64),
        };

        let mut sw = self.stop_watch();
        bar.tick();
        for &body_id in bodies {
            let module = body_id.module(db);
            let full_name = move || {
                module
                    .krate()
                    .display_name(db)
                    .map(|it| it.canonical_name().to_string())
                    .into_iter()
                    .chain(
                        module
                            .path_to_root(db)
                            .into_iter()
                            .filter_map(|it| it.name(db))
                            .rev()
                            .chain(Some(body_id.name(db).unwrap_or_else(Name::missing)))
                            .map(|it| it.display(db).to_string()),
                    )
                    .join("::")
            };
            if let Some(only_name) = self.only.as_deref() {
                if body_id.name(db).unwrap_or_else(Name::missing).display(db).to_string()
                    != only_name
                    && full_name() != only_name
                {
                    continue;
                }
            }
            let msg = move || {
                if verbosity.is_verbose() {
                    let source = match body_id {
                        DefWithBody::Function(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Static(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Const(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Variant(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::InTypeConst(_) => unimplemented!(),
                    };
                    if let Some(src) = source {
                        let original_file = src.file_id.original_file(db);
                        let path = vfs.file_path(original_file);
                        let syntax_range = src.value.text_range();
                        format!("processing: {} ({} {:?})", full_name(), path, syntax_range)
                    } else {
                        format!("processing: {}", full_name())
                    }
                } else {
                    format!("processing: {}", full_name())
                }
            };
            if verbosity.is_spammy() {
                bar.println(msg());
            }
            bar.set_message(msg);
            db.body_with_source_map(body_id.into());
            bar.inc(1);
        }

        bar.finish_and_clear();
        let body_lowering_time = sw.elapsed();
        eprintln!("{:<20} {}", "Body lowering:", body_lowering_time);
        report_metric("body lowering time", body_lowering_time.time.as_millis() as u64, "ms");
    }

    fn run_ide_things(&self, analysis: Analysis, mut file_ids: Vec<FileId>) {
        file_ids.sort();
        file_ids.dedup();
        let mut sw = self.stop_watch();

        for &file_id in &file_ids {
            _ = analysis.diagnostics(
                &DiagnosticsConfig {
                    enabled: true,
                    proc_macros_enabled: true,
                    proc_attr_macros_enabled: true,
                    disable_experimental: false,
                    disabled: Default::default(),
                    expr_fill_default: Default::default(),
                    insert_use: ide_db::imports::insert_use::InsertUseConfig {
                        granularity: ide_db::imports::insert_use::ImportGranularity::Crate,
                        enforce_granularity: true,
                        prefix_kind: hir::PrefixKind::ByCrate,
                        group: true,
                        skip_glob_imports: true,
                    },
                    prefer_no_std: Default::default(),
                },
                ide::AssistResolveStrategy::All,
                file_id,
            );
        }
        for &file_id in &file_ids {
            _ = analysis.inlay_hints(
                &InlayHintsConfig {
                    render_colons: false,
                    type_hints: true,
                    discriminant_hints: ide::DiscriminantHints::Always,
                    parameter_hints: true,
                    chaining_hints: true,
                    adjustment_hints: ide::AdjustmentHints::Always,
                    adjustment_hints_mode: ide::AdjustmentHintsMode::Postfix,
                    adjustment_hints_hide_outside_unsafe: false,
                    closure_return_type_hints: ide::ClosureReturnTypeHints::Always,
                    closure_capture_hints: true,
                    binding_mode_hints: true,
                    lifetime_elision_hints: ide::LifetimeElisionHints::Always,
                    param_names_for_lifetime_elision_hints: true,
                    hide_named_constructor_hints: false,
                    hide_closure_initialization_hints: false,
                    closure_style: hir::ClosureStyle::ImplFn,
                    max_length: Some(25),
                    closing_brace_hints_min_lines: Some(20),
                },
                file_id,
                None,
            );
        }
        for &file_id in &file_ids {
            analysis
                .annotations(
                    &AnnotationConfig {
                        binary_target: true,
                        annotate_runnables: true,
                        annotate_impls: true,
                        annotate_references: false,
                        annotate_method_references: false,
                        annotate_enum_variant_references: false,
                        location: ide::AnnotationLocation::AboveName,
                    },
                    file_id,
                )
                .unwrap()
                .into_iter()
                .for_each(|annotation| {
                    _ = analysis.resolve_annotation(annotation);
                });
        }
        let ide_time = sw.elapsed();
        eprintln!("{:<20} {} ({} files)", "IDE:", ide_time, file_ids.len());
    }

    fn stop_watch(&self) -> StopWatch {
        StopWatch::start().memory(self.memory_usage)
    }
}

fn location_csv_expr(db: &RootDatabase, vfs: &Vfs, sm: &BodySourceMap, expr_id: ExprId) -> String {
    let src = match sm.expr_syntax(expr_id) {
        Ok(s) => s,
        Err(SyntheticSyntax) => return "synthetic,,".to_string(),
    };
    let root = db.parse_or_expand(src.file_id);
    let node = src.map(|e| e.to_node(&root).syntax().clone());
    let original_range = node.as_ref().original_file_range(db);
    let path = vfs.file_path(original_range.file_id);
    let line_index = db.line_index(original_range.file_id);
    let text_range = original_range.range;
    let (start, end) =
        (line_index.line_col(text_range.start()), line_index.line_col(text_range.end()));
    format!("{path},{}:{},{}:{}", start.line + 1, start.col, end.line + 1, end.col)
}

fn location_csv_pat(db: &RootDatabase, vfs: &Vfs, sm: &BodySourceMap, pat_id: PatId) -> String {
    let src = match sm.pat_syntax(pat_id) {
        Ok(s) => s,
        Err(SyntheticSyntax) => return "synthetic,,".to_string(),
    };
    let root = db.parse_or_expand(src.file_id);
    let node = src.map(|e| {
        e.either(|it| it.to_node(&root).syntax().clone(), |it| it.to_node(&root).syntax().clone())
    });
    let original_range = node.as_ref().original_file_range(db);
    let path = vfs.file_path(original_range.file_id);
    let line_index = db.line_index(original_range.file_id);
    let text_range = original_range.range;
    let (start, end) =
        (line_index.line_col(text_range.start()), line_index.line_col(text_range.end()));
    format!("{path},{}:{},{}:{}", start.line + 1, start.col, end.line + 1, end.col)
}

fn expr_syntax_range(
    db: &RootDatabase,
    vfs: &Vfs,
    sm: &BodySourceMap,
    expr_id: ExprId,
) -> Option<(VfsPath, LineCol, LineCol)> {
    let src = sm.expr_syntax(expr_id);
    if let Ok(src) = src {
        let root = db.parse_or_expand(src.file_id);
        let node = src.map(|e| e.to_node(&root).syntax().clone());
        let original_range = node.as_ref().original_file_range(db);
        let path = vfs.file_path(original_range.file_id);
        let line_index = db.line_index(original_range.file_id);
        let text_range = original_range.range;
        let (start, end) =
            (line_index.line_col(text_range.start()), line_index.line_col(text_range.end()));
        Some((path, start, end))
    } else {
        None
    }
}
fn pat_syntax_range(
    db: &RootDatabase,
    vfs: &Vfs,
    sm: &BodySourceMap,
    pat_id: PatId,
) -> Option<(VfsPath, LineCol, LineCol)> {
    let src = sm.pat_syntax(pat_id);
    if let Ok(src) = src {
        let root = db.parse_or_expand(src.file_id);
        let node = src.map(|e| {
            e.either(
                |it| it.to_node(&root).syntax().clone(),
                |it| it.to_node(&root).syntax().clone(),
            )
        });
        let original_range = node.as_ref().original_file_range(db);
        let path = vfs.file_path(original_range.file_id);
        let line_index = db.line_index(original_range.file_id);
        let text_range = original_range.range;
        let (start, end) =
            (line_index.line_col(text_range.start()), line_index.line_col(text_range.end()));
        Some((path, start, end))
    } else {
        None
    }
}

fn shuffle<T>(rng: &mut Rand32, slice: &mut [T]) {
    for i in 0..slice.len() {
        randomize_first(rng, &mut slice[i..]);
    }

    fn randomize_first<T>(rng: &mut Rand32, slice: &mut [T]) {
        assert!(!slice.is_empty());
        let idx = rng.rand_range(0..slice.len() as u32) as usize;
        slice.swap(0, idx);
    }
}

fn percentage(n: u64, total: u64) -> u64 {
    (n * 100).checked_div(total).unwrap_or(100)
}

fn syntax_len(node: SyntaxNode) -> usize {
    // Macro expanded code doesn't contain whitespace, so erase *all* whitespace
    // to make macro and non-macro code comparable.
    node.to_string().replace(|it: char| it.is_ascii_whitespace(), "").len()
}
