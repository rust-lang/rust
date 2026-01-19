//! Fully type-check project and print various stats, like the number of type
//! errors.

use std::{
    env, fmt,
    ops::AddAssign,
    panic::{AssertUnwindSafe, catch_unwind},
    time::{SystemTime, UNIX_EPOCH},
};

use cfg::{CfgAtom, CfgDiff};
use hir::{
    Adt, AssocItem, Crate, DefWithBody, FindPathConfig, HasCrate, HasSource, HirDisplay, ModuleDef,
    Name, crate_lang_items,
    db::{DefDatabase, ExpandDatabase, HirDatabase},
    next_solver::{DbInterner, GenericArgs},
};
use hir_def::{
    SyntheticSyntax,
    expr_store::BodySourceMap,
    hir::{ExprId, PatId},
};
use hir_ty::InferenceResult;
use ide::{
    Analysis, AnalysisHost, AnnotationConfig, DiagnosticsConfig, Edition, InlayFieldsToResolve,
    InlayHintsConfig, LineCol, RootDatabase,
};
use ide_db::{
    EditionedFileId, LineIndexDatabase, MiniCore, SnippetCap,
    base_db::{SourceDatabase, salsa::Database},
};
use itertools::Itertools;
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace};
use oorandom::Rand32;
use profile::StopWatch;
use project_model::{CargoConfig, CfgOverrides, ProjectManifest, ProjectWorkspace, RustLibSource};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::inherent::Ty as _;
use syntax::AstNode;
use vfs::{AbsPathBuf, Vfs, VfsPath};

use crate::cli::{
    Verbosity,
    flags::{self, OutputFormat},
    full_name_of_item, print_memory_usage,
    progress_report::ProgressReport,
    report_metric,
};

impl flags::AnalysisStats {
    pub fn run(self, verbosity: Verbosity) -> anyhow::Result<()> {
        let mut rng = {
            let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
            Rand32::new(seed)
        };

        let cargo_config = CargoConfig {
            sysroot: match self.no_sysroot {
                true => None,
                false => Some(RustLibSource::Discover),
            },
            all_targets: true,
            set_test: !self.no_test,
            cfg_overrides: CfgOverrides {
                global: CfgDiff::new(vec![CfgAtom::Flag(hir::sym::miri)], vec![]),
                selective: Default::default(),
            },
            ..Default::default()
        };
        let no_progress = &|_| ();

        let mut db_load_sw = self.stop_watch();

        let path = AbsPathBuf::assert_utf8(env::current_dir()?.join(&self.path));
        let manifest = ProjectManifest::discover_single(&path)?;

        let mut workspace = ProjectWorkspace::load(manifest, &cargo_config, no_progress)?;
        let metadata_time = db_load_sw.elapsed();
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: !self.disable_build_scripts,
            with_proc_macro_server: if self.disable_proc_macros {
                ProcMacroServerChoice::None
            } else {
                match self.proc_macro_srv {
                    Some(ref path) => {
                        let path = vfs::AbsPathBuf::assert_utf8(path.to_owned());
                        ProcMacroServerChoice::Explicit(path)
                    }
                    None => ProcMacroServerChoice::Sysroot,
                }
            },
            prefill_caches: false,
            proc_macro_processes: 1,
        };

        let build_scripts_time = if self.disable_build_scripts {
            None
        } else {
            let mut build_scripts_sw = self.stop_watch();
            let bs = workspace.run_build_scripts(&cargo_config, no_progress)?;
            workspace.set_build_scripts(bs);
            Some(build_scripts_sw.elapsed())
        };

        let (db, vfs, _proc_macro) =
            load_workspace(workspace.clone(), &cargo_config.extra_env, &load_cargo_config)?;
        eprint!("{:<20} {}", "Database loaded:", db_load_sw.elapsed());
        eprint!(" (metadata {metadata_time}");
        if let Some(build_scripts_time) = build_scripts_time {
            eprint!("; build {build_scripts_time}");
        }
        eprintln!(")");

        let mut host = AnalysisHost::with_database(db);
        let db = host.raw_database();

        let mut analysis_sw = self.stop_watch();

        let mut krates = Crate::all(db);
        if self.randomize {
            shuffle(&mut rng, &mut krates);
        }

        let mut item_tree_sw = self.stop_watch();
        let source_roots = krates
            .iter()
            .cloned()
            .map(|krate| db.file_source_root(krate.root_file(db)).source_root_id(db))
            .unique();

        let mut dep_loc = 0;
        let mut workspace_loc = 0;
        let mut dep_item_trees = 0;
        let mut workspace_item_trees = 0;

        let mut workspace_item_stats = PrettyItemStats::default();
        let mut dep_item_stats = PrettyItemStats::default();

        for source_root_id in source_roots {
            let source_root = db.source_root(source_root_id).source_root(db);
            for file_id in source_root.iter() {
                if let Some(p) = source_root.path_for_file(&file_id)
                    && let Some((_, Some("rs"))) = p.name_and_extension()
                {
                    // measure workspace/project code
                    if !source_root.is_library || self.with_deps {
                        let length = db.file_text(file_id).text(db).lines().count();
                        let item_stats = db
                            .file_item_tree(
                                EditionedFileId::current_edition_guess_origin(db, file_id).into(),
                            )
                            .item_tree_stats()
                            .into();

                        workspace_loc += length;
                        workspace_item_trees += 1;
                        workspace_item_stats += item_stats;
                    } else {
                        let length = db.file_text(file_id).text(db).lines().count();
                        let item_stats = db
                            .file_item_tree(
                                EditionedFileId::current_edition_guess_origin(db, file_id).into(),
                            )
                            .item_tree_stats()
                            .into();

                        dep_loc += length;
                        dep_item_trees += 1;
                        dep_item_stats += item_stats;
                    }
                }
            }
        }
        eprintln!("  item trees: {workspace_item_trees}");
        let item_tree_time = item_tree_sw.elapsed();

        eprintln!(
            "  dependency lines of code: {}, item trees: {}",
            UsizeWithUnderscore(dep_loc),
            UsizeWithUnderscore(dep_item_trees),
        );
        eprintln!("  dependency item stats: {dep_item_stats}");

        // FIXME(salsa-transition): bring back stats for ParseQuery (file size)
        // and ParseMacroExpansionQuery (macro expansion "file") size whenever we implement
        // Salsa's memory usage tracking works with tracked functions.

        // let mut total_file_size = Bytes::default();
        // for e in ide_db::base_db::ParseQuery.in_db(db).entries::<Vec<_>>() {
        //     total_file_size += syntax_len(db.parse(e.key).syntax_node())
        // }

        // let mut total_macro_file_size = Bytes::default();
        // for e in hir::db::ParseMacroExpansionQuery.in_db(db).entries::<Vec<_>>() {
        //     let val = db.parse_macro_expansion(e.key).value.0;
        //     total_macro_file_size += syntax_len(val.syntax_node())
        // }
        // eprintln!("source files: {total_file_size}, macro files: {total_macro_file_size}");

        eprintln!("{:<20} {}", "Item Tree Collection:", item_tree_time);
        report_metric("item tree time", item_tree_time.time.as_millis() as u64, "ms");
        eprintln!("  Total Statistics:");

        let mut crate_def_map_sw = self.stop_watch();
        let mut num_crates = 0;
        let mut visited_modules = FxHashSet::default();
        let mut visit_queue = Vec::new();
        for &krate in &krates {
            let module = krate.root_module(db);
            let file_id = module.definition_source_file_id(db);
            let file_id = file_id.original_file(db);

            let source_root = db.file_source_root(file_id.file_id(db)).source_root_id(db);
            let source_root = db.source_root(source_root).source_root(db);
            if !source_root.is_library || self.with_deps {
                num_crates += 1;
                visit_queue.push(module);
            }
        }

        if self.randomize {
            shuffle(&mut rng, &mut visit_queue);
        }

        eprint!("    crates: {num_crates}");
        let mut num_decls = 0;
        let mut bodies = Vec::new();
        let mut adts = Vec::new();
        let mut file_ids = Vec::new();

        let mut num_traits = 0;
        let mut num_macro_rules_macros = 0;
        let mut num_proc_macros = 0;

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
                        }
                        ModuleDef::Static(s) => bodies.push(DefWithBody::from(s)),
                        ModuleDef::Trait(_) => num_traits += 1,
                        ModuleDef::Macro(m) => match m.kind(db) {
                            hir::MacroKind::Declarative => num_macro_rules_macros += 1,
                            hir::MacroKind::Derive
                            | hir::MacroKind::Attr
                            | hir::MacroKind::ProcMacro => num_proc_macros += 1,
                            _ => (),
                        },
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
            bodies
                .iter()
                .filter(|it| matches!(it, DefWithBody::Const(_) | DefWithBody::Static(_)))
                .count(),
        );

        eprintln!("  Workspace:");
        eprintln!(
            "    traits: {num_traits}, macro_rules macros: {num_macro_rules_macros}, proc_macros: {num_proc_macros}"
        );
        eprintln!(
            "    lines of code: {}, item trees: {}",
            UsizeWithUnderscore(workspace_loc),
            UsizeWithUnderscore(workspace_item_trees),
        );
        eprintln!("    usages: {workspace_item_stats}");

        eprintln!("  Dependencies:");
        eprintln!(
            "    lines of code: {}, item trees: {}",
            UsizeWithUnderscore(dep_loc),
            UsizeWithUnderscore(dep_item_trees),
        );
        eprintln!("    declarations: {dep_item_stats}");

        let crate_def_map_time = crate_def_map_sw.elapsed();
        eprintln!("{:<20} {}", "Item Collection:", crate_def_map_time);
        report_metric("crate def map time", crate_def_map_time.time.as_millis() as u64, "ms");

        if self.randomize {
            shuffle(&mut rng, &mut bodies);
        }

        hir::attach_db(db, || {
            if !self.skip_lang_items {
                self.run_lang_items(db, &krates, verbosity);
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
                self.run_const_eval(db, &bodies, verbosity);
            }
        });

        file_ids.sort();
        file_ids.dedup();

        if self.run_all_ide_things {
            self.run_ide_things(host.analysis(), &file_ids, db, &vfs, verbosity);
        }

        if self.run_term_search {
            self.run_term_search(&workspace, db, &vfs, &file_ids, verbosity);
        }

        let db = host.raw_database_mut();
        db.trigger_lru_eviction();
        hir::clear_tls_solver_cache();
        unsafe { hir::collect_ty_garbage() };

        let total_span = analysis_sw.elapsed();
        eprintln!("{:<20} {total_span}", "Total:");
        report_metric("total time", total_span.time.as_millis() as u64, "ms");
        if let Some(instructions) = total_span.instructions {
            report_metric("total instructions", instructions, "#instr");
        }
        report_metric("total memory", total_span.memory.allocated.megabytes() as u64, "MB");

        if verbosity.is_verbose() {
            print_memory_usage(host, vfs);
        }

        Ok(())
    }

    fn run_data_layout(&self, db: &RootDatabase, adts: &[hir::Adt], verbosity: Verbosity) {
        let mut sw = self.stop_watch();
        let mut all = 0;
        let mut fail = 0;
        for &a in adts {
            let interner = DbInterner::new_no_crate(db);
            let generic_params = db.generic_params(a.into());
            if generic_params.iter_type_or_consts().next().is_some()
                || generic_params.iter_lt().next().is_some()
            {
                // Data types with generics don't have layout.
                continue;
            }
            all += 1;
            let Err(e) = db.layout_of_adt(
                hir_def::AdtId::from(a),
                GenericArgs::empty(interner).store(),
                hir_ty::ParamEnvAndCrate {
                    param_env: db.trait_environment(a.into()),
                    krate: a.krate(db).into(),
                }
                .store(),
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

    fn run_const_eval(&self, db: &RootDatabase, bodies: &[DefWithBody], verbosity: Verbosity) {
        let len = bodies
            .iter()
            .filter(|body| matches!(body, DefWithBody::Const(_) | DefWithBody::Static(_)))
            .count();
        let mut bar = match verbosity {
            Verbosity::Quiet | Verbosity::Spammy => ProgressReport::hidden(),
            _ if self.parallel || self.output.is_some() => ProgressReport::hidden(),
            _ => ProgressReport::new(len),
        };

        let mut sw = self.stop_watch();
        let mut all = 0;
        let mut fail = 0;
        for &b in bodies {
            bar.set_message(move || format!("const eval: {}", full_name(db, b, b.module(db))));
            let res = match b {
                DefWithBody::Const(c) => c.eval(db),
                DefWithBody::Static(s) => s.eval(db),
                _ => continue,
            };
            bar.inc(1);
            all += 1;
            let Err(error) = res else {
                continue;
            };
            if verbosity.is_spammy() {
                let full_name =
                    full_name_of_item(db, b.module(db), b.name(db).unwrap_or(Name::missing()));
                bar.println(format!("Const eval for {full_name} failed due {error:?}"));
            }
            fail += 1;
        }
        bar.finish_and_clear();
        let const_eval_time = sw.elapsed();
        eprintln!("{:<20} {}", "Const evaluation:", const_eval_time);
        eprintln!("Failed const evals: {fail} ({}%)", percentage(fail, all));
        report_metric("failed const evals", fail, "#");
        report_metric("const eval time", const_eval_time.time.as_millis() as u64, "ms");
    }

    /// Invariant: `file_ids` must be sorted and deduped before passing into here
    fn run_term_search(
        &self,
        ws: &ProjectWorkspace,
        db: &RootDatabase,
        vfs: &Vfs,
        file_ids: &[EditionedFileId],
        verbosity: Verbosity,
    ) {
        let cargo_config = CargoConfig {
            sysroot: match self.no_sysroot {
                true => None,
                false => Some(RustLibSource::Discover),
            },
            all_targets: true,
            ..Default::default()
        };

        let mut bar = match verbosity {
            Verbosity::Quiet | Verbosity::Spammy => ProgressReport::hidden(),
            _ if self.parallel || self.output.is_some() => ProgressReport::hidden(),
            _ => ProgressReport::new(file_ids.len()),
        };

        #[derive(Debug, Default)]
        struct Acc {
            tail_expr_syntax_hits: u64,
            tail_expr_no_term: u64,
            total_tail_exprs: u64,
            error_codes: FxHashMap<String, u32>,
            syntax_errors: u32,
        }

        let mut acc: Acc = Default::default();
        bar.tick();
        let mut sw = self.stop_watch();

        for &file_id in file_ids {
            let file_id = file_id.editioned_file_id(db);
            let sema = hir::Semantics::new(db);
            let display_target = match sema.first_crate(file_id.file_id()) {
                Some(krate) => krate.to_display_target(sema.db),
                None => continue,
            };

            let parse = sema.parse_guess_edition(file_id.into());
            let file_txt = db.file_text(file_id.into());
            let path = vfs.file_path(file_id.into()).as_path().unwrap();

            for node in parse.syntax().descendants() {
                let expr = match syntax::ast::Expr::cast(node.clone()) {
                    Some(it) => it,
                    None => continue,
                };
                let block = match syntax::ast::BlockExpr::cast(expr.syntax().clone()) {
                    Some(it) => it,
                    None => continue,
                };
                let target_ty = match sema.type_of_expr(&expr) {
                    Some(it) => it.adjusted(),
                    None => continue, // Failed to infer type
                };

                let expected_tail = match block.tail_expr() {
                    Some(it) => it,
                    None => continue,
                };

                if expected_tail.is_block_like() {
                    continue;
                }

                let range = sema.original_range(expected_tail.syntax()).range;
                let original_text: String = db
                    .file_text(file_id.into())
                    .text(db)
                    .chars()
                    .skip(usize::from(range.start()))
                    .take(usize::from(range.end()) - usize::from(range.start()))
                    .collect();

                let scope = match sema.scope(expected_tail.syntax()) {
                    Some(it) => it,
                    None => continue,
                };

                let ctx = hir::term_search::TermSearchCtx {
                    sema: &sema,
                    scope: &scope,
                    goal: target_ty,
                    config: hir::term_search::TermSearchConfig {
                        enable_borrowcheck: true,
                        ..Default::default()
                    },
                };
                let found_terms = hir::term_search::term_search(&ctx);

                if found_terms.is_empty() {
                    acc.tail_expr_no_term += 1;
                    acc.total_tail_exprs += 1;
                    // println!("\n{original_text}\n");
                    continue;
                };

                fn trim(s: &str) -> String {
                    s.chars().filter(|c| !c.is_whitespace()).collect()
                }

                let todo = syntax::ast::make::ext::expr_todo().to_string();
                let mut formatter = |_: &hir::Type<'_>| todo.clone();
                let mut syntax_hit_found = false;
                for term in found_terms {
                    let generated = term
                        .gen_source_code(
                            &scope,
                            &mut formatter,
                            FindPathConfig {
                                prefer_no_std: false,
                                prefer_prelude: true,
                                prefer_absolute: false,
                                allow_unstable: true,
                            },
                            display_target,
                        )
                        .unwrap();
                    syntax_hit_found |= trim(&original_text) == trim(&generated);

                    // Validate if type-checks
                    let mut txt = file_txt.text(db).to_string();

                    let edit = ide::TextEdit::replace(range, generated.clone());
                    edit.apply(&mut txt);

                    if self.validate_term_search {
                        std::fs::write(path, txt).unwrap();

                        let res = ws.run_build_scripts(&cargo_config, &|_| ()).unwrap();
                        if let Some(err) = res.error()
                            && err.contains("error: could not compile")
                        {
                            if let Some(mut err_idx) = err.find("error[E") {
                                err_idx += 7;
                                let err_code = &err[err_idx..err_idx + 4];
                                match err_code {
                                    "0282" | "0283" => continue, // Byproduct of testing method
                                    "0277" | "0308" if generated.contains(&todo) => continue, // See https://github.com/rust-lang/rust/issues/69882
                                    // FIXME: In some rare cases `AssocItem::container_or_implemented_trait` returns `None` for trait methods.
                                    // Generated code is valid in case traits are imported
                                    "0599"
                                        if err.contains(
                                            "the following trait is implemented but not in scope",
                                        ) =>
                                    {
                                        continue;
                                    }
                                    _ => (),
                                }
                                bar.println(err);
                                bar.println(generated);
                                acc.error_codes
                                    .entry(err_code.to_owned())
                                    .and_modify(|n| *n += 1)
                                    .or_insert(1);
                            } else {
                                acc.syntax_errors += 1;
                                bar.println(format!("Syntax error: \n{err}"));
                            }
                        }
                    }
                }

                if syntax_hit_found {
                    acc.tail_expr_syntax_hits += 1;
                }
                acc.total_tail_exprs += 1;

                let msg = move || {
                    format!(
                        "processing: {:<50}",
                        trim(&original_text).chars().take(50).collect::<String>()
                    )
                };
                if verbosity.is_spammy() {
                    bar.println(msg());
                }
                bar.set_message(msg);
            }
            // Revert file back to original state
            if self.validate_term_search {
                std::fs::write(path, file_txt.text(db).to_string()).unwrap();
            }

            bar.inc(1);
        }
        let term_search_time = sw.elapsed();

        bar.println(format!(
            "Tail Expr syntactic hits: {}/{} ({}%)",
            acc.tail_expr_syntax_hits,
            acc.total_tail_exprs,
            percentage(acc.tail_expr_syntax_hits, acc.total_tail_exprs)
        ));
        bar.println(format!(
            "Tail Exprs found: {}/{} ({}%)",
            acc.total_tail_exprs - acc.tail_expr_no_term,
            acc.total_tail_exprs,
            percentage(acc.total_tail_exprs - acc.tail_expr_no_term, acc.total_tail_exprs)
        ));
        if self.validate_term_search {
            bar.println(format!(
                "Tail Exprs total errors: {}, syntax errors: {}, error codes:",
                acc.error_codes.values().sum::<u32>() + acc.syntax_errors,
                acc.syntax_errors,
            ));
            for (err, count) in acc.error_codes {
                bar.println(format!(
                    "    E{err}: {count:>5}  (https://doc.rust-lang.org/error_codes/E{err}.html)"
                ));
            }
        }
        bar.println(format!(
            "Term search avg time: {}ms",
            term_search_time.time.as_millis() as u64 / acc.total_tail_exprs
        ));
        bar.println(format!("{:<20} {}", "Term search:", term_search_time));
        report_metric("term search time", term_search_time.time.as_millis() as u64, "ms");

        bar.finish_and_clear();
    }

    fn run_mir_lowering(&self, db: &RootDatabase, bodies: &[DefWithBody], verbosity: Verbosity) {
        let mut bar = match verbosity {
            Verbosity::Quiet | Verbosity::Spammy => ProgressReport::hidden(),
            _ if self.parallel || self.output.is_some() => ProgressReport::hidden(),
            _ => ProgressReport::new(bodies.len()),
        };
        let mut sw = self.stop_watch();
        let mut all = 0;
        let mut fail = 0;
        for &body in bodies {
            bar.set_message(move || {
                format!("mir lowering: {}", full_name(db, body, body.module(db)))
            });
            bar.inc(1);
            if matches!(body, DefWithBody::Variant(_)) {
                continue;
            }
            let module = body.module(db);
            if !self.should_process(db, body, module) {
                continue;
            }

            all += 1;
            let Ok(body_id) = body.try_into() else {
                continue;
            };
            let Err(e) = db.mir_body(body_id) else {
                continue;
            };
            if verbosity.is_spammy() {
                let full_name = module
                    .path_to_root(db)
                    .into_iter()
                    .rev()
                    .filter_map(|it| it.name(db))
                    .chain(Some(body.name(db).unwrap_or_else(Name::missing)))
                    .map(|it| it.display(db, Edition::LATEST).to_string())
                    .join("::");
                bar.println(format!("Mir body for {full_name} failed due {e:?}"));
            }
            fail += 1;
            bar.tick();
        }
        let mir_lowering_time = sw.elapsed();
        bar.finish_and_clear();
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
            _ => ProgressReport::new(bodies.len()),
        };

        if self.parallel {
            let mut inference_sw = self.stop_watch();
            let bodies = bodies.iter().filter_map(|&body| body.try_into().ok()).collect::<Vec<_>>();
            bodies
                .par_iter()
                .map_with(db.clone(), |snap, &body| {
                    snap.body(body);
                    InferenceResult::for_body(snap, body);
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
        let mut panics = 0;
        for &body_id in bodies {
            let Ok(body_def_id) = body_id.try_into() else { continue };
            let name = body_id.name(db).unwrap_or_else(Name::missing);
            let module = body_id.module(db);
            let display_target = module.krate(db).to_display_target(db);
            if let Some(only_name) = self.only.as_deref()
                && name.display(db, Edition::LATEST).to_string() != only_name
                && full_name(db, body_id, module) != only_name
            {
                continue;
            }
            let msg = move || {
                if verbosity.is_verbose() {
                    let source = match body_id {
                        DefWithBody::Function(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Static(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Const(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Variant(it) => it.source(db).map(|it| it.syntax().cloned()),
                    };
                    if let Some(src) = source {
                        let original_file = src.file_id.original_file(db);
                        let path = vfs.file_path(original_file.file_id(db));
                        let syntax_range = src.text_range();
                        format!(
                            "processing: {} ({} {:?})",
                            full_name(db, body_id, module),
                            path,
                            syntax_range
                        )
                    } else {
                        format!("processing: {}", full_name(db, body_id, module))
                    }
                } else {
                    format!("processing: {}", full_name(db, body_id, module))
                }
            };
            if verbosity.is_spammy() {
                bar.println(msg());
            }
            bar.set_message(msg);
            let body = db.body(body_def_id);
            let inference_result =
                catch_unwind(AssertUnwindSafe(|| InferenceResult::for_body(db, body_def_id)));
            let inference_result = match inference_result {
                Ok(inference_result) => inference_result,
                Err(p) => {
                    if let Some(s) = p.downcast_ref::<&str>() {
                        eprintln!("infer panicked for {}: {}", full_name(db, body_id, module), s);
                    } else if let Some(s) = p.downcast_ref::<String>() {
                        eprintln!("infer panicked for {}: {}", full_name(db, body_id, module), s);
                    } else {
                        eprintln!("infer panicked for {}", full_name(db, body_id, module));
                    }
                    panics += 1;
                    bar.inc(1);
                    continue;
                }
            };
            // This query is LRU'd, so actually calling it will skew the timing results.
            let sm = || db.body_with_source_map(body_def_id).1;

            // region:expressions
            let (previous_exprs, previous_unknown, previous_partially_unknown) =
                (num_exprs, num_exprs_unknown, num_exprs_partially_unknown);
            for (expr_id, _) in body.exprs() {
                let ty = inference_result.expr_ty(expr_id);
                num_exprs += 1;
                let unknown_or_partial = if ty.is_ty_error() {
                    num_exprs_unknown += 1;
                    if verbosity.is_spammy() {
                        if let Some((path, start, end)) = expr_syntax_range(db, vfs, &sm(), expr_id)
                        {
                            bar.println(format!(
                                "{} {}:{}-{}:{}: Unknown type",
                                path,
                                start.line + 1,
                                start.col,
                                end.line + 1,
                                end.col,
                            ));
                        } else {
                            bar.println(format!(
                                "{}: Unknown type",
                                name.display(db, Edition::LATEST)
                            ));
                        }
                    }
                    true
                } else {
                    let is_partially_unknown = ty.references_non_lt_error();
                    if is_partially_unknown {
                        num_exprs_partially_unknown += 1;
                    }
                    is_partially_unknown
                };
                if self.only.is_some() && verbosity.is_spammy() {
                    // in super-verbose mode for just one function, we print every single expression
                    if let Some((_, start, end)) = expr_syntax_range(db, vfs, &sm(), expr_id) {
                        bar.println(format!(
                            "{}:{}-{}:{}: {}",
                            start.line + 1,
                            start.col,
                            end.line + 1,
                            end.col,
                            ty.display(db, display_target)
                        ));
                    } else {
                        bar.println(format!(
                            "unknown location: {}",
                            ty.display(db, display_target)
                        ));
                    }
                }
                if unknown_or_partial && self.output == Some(OutputFormat::Csv) {
                    println!(
                        r#"{},type,"{}""#,
                        location_csv_expr(db, vfs, &sm(), expr_id),
                        ty.display(db, display_target)
                    );
                }
                if let Some(mismatch) = inference_result.type_mismatch_for_expr(expr_id) {
                    num_expr_type_mismatches += 1;
                    if verbosity.is_verbose() {
                        if let Some((path, start, end)) = expr_syntax_range(db, vfs, &sm(), expr_id)
                        {
                            bar.println(format!(
                                "{} {}:{}-{}:{}: Expected {}, got {}",
                                path,
                                start.line + 1,
                                start.col,
                                end.line + 1,
                                end.col,
                                mismatch.expected.as_ref().display(db, display_target),
                                mismatch.actual.as_ref().display(db, display_target)
                            ));
                        } else {
                            bar.println(format!(
                                "{}: Expected {}, got {}",
                                name.display(db, Edition::LATEST),
                                mismatch.expected.as_ref().display(db, display_target),
                                mismatch.actual.as_ref().display(db, display_target)
                            ));
                        }
                    }
                    if self.output == Some(OutputFormat::Csv) {
                        println!(
                            r#"{},mismatch,"{}","{}""#,
                            location_csv_expr(db, vfs, &sm(), expr_id),
                            mismatch.expected.as_ref().display(db, display_target),
                            mismatch.actual.as_ref().display(db, display_target)
                        );
                    }
                }
            }
            if verbosity.is_spammy() {
                bar.println(format!(
                    "In {}: {} exprs, {} unknown, {} partial",
                    full_name(db, body_id, module),
                    num_exprs - previous_exprs,
                    num_exprs_unknown - previous_unknown,
                    num_exprs_partially_unknown - previous_partially_unknown
                ));
            }
            // endregion:expressions

            // region:patterns
            let (previous_pats, previous_unknown, previous_partially_unknown) =
                (num_pats, num_pats_unknown, num_pats_partially_unknown);
            for (pat_id, _) in body.pats() {
                let ty = inference_result.pat_ty(pat_id);
                num_pats += 1;
                let unknown_or_partial = if ty.is_ty_error() {
                    num_pats_unknown += 1;
                    if verbosity.is_spammy() {
                        if let Some((path, start, end)) = pat_syntax_range(db, vfs, &sm(), pat_id) {
                            bar.println(format!(
                                "{} {}:{}-{}:{}: Unknown type",
                                path,
                                start.line + 1,
                                start.col,
                                end.line + 1,
                                end.col,
                            ));
                        } else {
                            bar.println(format!(
                                "{}: Unknown type",
                                name.display(db, Edition::LATEST)
                            ));
                        }
                    }
                    true
                } else {
                    let is_partially_unknown = ty.references_non_lt_error();
                    if is_partially_unknown {
                        num_pats_partially_unknown += 1;
                    }
                    is_partially_unknown
                };
                if self.only.is_some() && verbosity.is_spammy() {
                    // in super-verbose mode for just one function, we print every single pattern
                    if let Some((_, start, end)) = pat_syntax_range(db, vfs, &sm(), pat_id) {
                        bar.println(format!(
                            "{}:{}-{}:{}: {}",
                            start.line + 1,
                            start.col,
                            end.line + 1,
                            end.col,
                            ty.display(db, display_target)
                        ));
                    } else {
                        bar.println(format!(
                            "unknown location: {}",
                            ty.display(db, display_target)
                        ));
                    }
                }
                if unknown_or_partial && self.output == Some(OutputFormat::Csv) {
                    println!(
                        r#"{},type,"{}""#,
                        location_csv_pat(db, vfs, &sm(), pat_id),
                        ty.display(db, display_target)
                    );
                }
                if let Some(mismatch) = inference_result.type_mismatch_for_pat(pat_id) {
                    num_pat_type_mismatches += 1;
                    if verbosity.is_verbose() {
                        if let Some((path, start, end)) = pat_syntax_range(db, vfs, &sm(), pat_id) {
                            bar.println(format!(
                                "{} {}:{}-{}:{}: Expected {}, got {}",
                                path,
                                start.line + 1,
                                start.col,
                                end.line + 1,
                                end.col,
                                mismatch.expected.as_ref().display(db, display_target),
                                mismatch.actual.as_ref().display(db, display_target)
                            ));
                        } else {
                            bar.println(format!(
                                "{}: Expected {}, got {}",
                                name.display(db, Edition::LATEST),
                                mismatch.expected.as_ref().display(db, display_target),
                                mismatch.actual.as_ref().display(db, display_target)
                            ));
                        }
                    }
                    if self.output == Some(OutputFormat::Csv) {
                        println!(
                            r#"{},mismatch,"{}","{}""#,
                            location_csv_pat(db, vfs, &sm(), pat_id),
                            mismatch.expected.as_ref().display(db, display_target),
                            mismatch.actual.as_ref().display(db, display_target)
                        );
                    }
                }
            }
            if verbosity.is_spammy() {
                bar.println(format!(
                    "In {}: {} pats, {} unknown, {} partial",
                    full_name(db, body_id, module),
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
        eprintln!("  panics: {panics}");
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
            _ => ProgressReport::new(bodies.len()),
        };

        let mut sw = self.stop_watch();
        bar.tick();
        for &body_id in bodies {
            let Ok(body_def_id) = body_id.try_into() else { continue };
            let module = body_id.module(db);
            if !self.should_process(db, body_id, module) {
                continue;
            }
            let msg = move || {
                if verbosity.is_verbose() {
                    let source = match body_id {
                        DefWithBody::Function(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Static(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Const(it) => it.source(db).map(|it| it.syntax().cloned()),
                        DefWithBody::Variant(it) => it.source(db).map(|it| it.syntax().cloned()),
                    };
                    if let Some(src) = source {
                        let original_file = src.file_id.original_file(db);
                        let path = vfs.file_path(original_file.file_id(db));
                        let syntax_range = src.text_range();
                        format!(
                            "processing: {} ({} {:?})",
                            full_name(db, body_id, module),
                            path,
                            syntax_range
                        )
                    } else {
                        format!("processing: {}", full_name(db, body_id, module))
                    }
                } else {
                    format!("processing: {}", full_name(db, body_id, module))
                }
            };
            if verbosity.is_spammy() {
                bar.println(msg());
            }
            bar.set_message(msg);
            db.body(body_def_id);
            bar.inc(1);
        }

        bar.finish_and_clear();
        let body_lowering_time = sw.elapsed();
        eprintln!("{:<20} {}", "Body lowering:", body_lowering_time);
        report_metric("body lowering time", body_lowering_time.time.as_millis() as u64, "ms");
    }

    fn run_lang_items(&self, db: &RootDatabase, crates: &[Crate], verbosity: Verbosity) {
        let mut bar = match verbosity {
            Verbosity::Quiet | Verbosity::Spammy => ProgressReport::hidden(),
            _ if self.output.is_some() => ProgressReport::hidden(),
            _ => ProgressReport::new(crates.len()),
        };

        let mut sw = self.stop_watch();
        bar.tick();
        for &krate in crates {
            crate_lang_items(db, krate.into());
            bar.inc(1);
        }

        bar.finish_and_clear();
        let time = sw.elapsed();
        eprintln!("{:<20} {}", "Crate lang items:", time);
        report_metric("crate lang items time", time.time.as_millis() as u64, "ms");
    }

    /// Invariant: `file_ids` must be sorted and deduped before passing into here
    fn run_ide_things(
        &self,
        analysis: Analysis,
        file_ids: &[EditionedFileId],
        db: &RootDatabase,
        vfs: &Vfs,
        verbosity: Verbosity,
    ) {
        let len = file_ids.len();
        let create_bar = || match verbosity {
            Verbosity::Quiet | Verbosity::Spammy => ProgressReport::hidden(),
            _ if self.parallel || self.output.is_some() => ProgressReport::hidden(),
            _ => ProgressReport::new(len),
        };

        let mut sw = self.stop_watch();

        let mut bar = create_bar();
        for &file_id in file_ids {
            let msg = format!("diagnostics: {}", vfs.file_path(file_id.file_id(db)));
            bar.set_message(move || msg.clone());
            _ = analysis.full_diagnostics(
                &DiagnosticsConfig {
                    enabled: true,
                    proc_macros_enabled: true,
                    proc_attr_macros_enabled: true,
                    disable_experimental: false,
                    disabled: Default::default(),
                    expr_fill_default: Default::default(),
                    snippet_cap: SnippetCap::new(true),
                    insert_use: ide_db::imports::insert_use::InsertUseConfig {
                        granularity: ide_db::imports::insert_use::ImportGranularity::Crate,
                        enforce_granularity: true,
                        prefix_kind: hir::PrefixKind::ByCrate,
                        group: true,
                        skip_glob_imports: true,
                    },
                    prefer_no_std: false,
                    prefer_prelude: true,
                    prefer_absolute: false,
                    style_lints: false,
                    term_search_fuel: 400,
                    term_search_borrowck: true,
                    show_rename_conflicts: true,
                },
                ide::AssistResolveStrategy::All,
                analysis.editioned_file_id_to_vfs(file_id),
            );
            bar.inc(1);
        }
        bar.finish_and_clear();

        let mut bar = create_bar();
        for &file_id in file_ids {
            let msg = format!("inlay hints: {}", vfs.file_path(file_id.file_id(db)));
            bar.set_message(move || msg.clone());
            _ = analysis.inlay_hints(
                &InlayHintsConfig {
                    render_colons: false,
                    type_hints: true,
                    sized_bound: false,
                    discriminant_hints: ide::DiscriminantHints::Always,
                    parameter_hints: true,
                    parameter_hints_for_missing_arguments: false,
                    generic_parameter_hints: ide::GenericParameterHints {
                        type_hints: true,
                        lifetime_hints: true,
                        const_hints: true,
                    },
                    chaining_hints: true,
                    adjustment_hints: ide::AdjustmentHints::Always,
                    adjustment_hints_disable_reborrows: true,
                    adjustment_hints_mode: ide::AdjustmentHintsMode::Postfix,
                    adjustment_hints_hide_outside_unsafe: false,
                    closure_return_type_hints: ide::ClosureReturnTypeHints::Always,
                    closure_capture_hints: true,
                    binding_mode_hints: true,
                    implicit_drop_hints: true,
                    implied_dyn_trait_hints: true,
                    lifetime_elision_hints: ide::LifetimeElisionHints::Always,
                    param_names_for_lifetime_elision_hints: true,
                    hide_inferred_type_hints: false,
                    hide_named_constructor_hints: false,
                    hide_closure_initialization_hints: false,
                    hide_closure_parameter_hints: false,
                    closure_style: hir::ClosureStyle::ImplFn,
                    max_length: Some(25),
                    closing_brace_hints_min_lines: Some(20),
                    fields_to_resolve: InlayFieldsToResolve::empty(),
                    range_exclusive_hints: true,
                    minicore: MiniCore::default(),
                },
                analysis.editioned_file_id_to_vfs(file_id),
                None,
            );
            bar.inc(1);
        }
        bar.finish_and_clear();

        let mut bar = create_bar();
        let annotation_config = AnnotationConfig {
            binary_target: true,
            annotate_runnables: true,
            annotate_impls: true,
            annotate_references: false,
            annotate_method_references: false,
            annotate_enum_variant_references: false,
            location: ide::AnnotationLocation::AboveName,
            filter_adjacent_derive_implementations: false,
            minicore: MiniCore::default(),
        };
        for &file_id in file_ids {
            let msg = format!("annotations: {}", vfs.file_path(file_id.file_id(db)));
            bar.set_message(move || msg.clone());
            analysis
                .annotations(&annotation_config, analysis.editioned_file_id_to_vfs(file_id))
                .unwrap()
                .into_iter()
                .for_each(|annotation| {
                    _ = analysis.resolve_annotation(&annotation_config, annotation);
                });
            bar.inc(1);
        }
        bar.finish_and_clear();

        let ide_time = sw.elapsed();
        eprintln!("{:<20} {} ({} files)", "IDE:", ide_time, file_ids.len());
    }

    fn should_process(&self, db: &RootDatabase, body_id: DefWithBody, module: hir::Module) -> bool {
        if let Some(only_name) = self.only.as_deref() {
            let name = body_id.name(db).unwrap_or_else(Name::missing);

            if name.display(db, Edition::LATEST).to_string() != only_name
                && full_name(db, body_id, module) != only_name
            {
                return false;
            }
        }
        true
    }

    fn stop_watch(&self) -> StopWatch {
        StopWatch::start()
    }
}

fn full_name(db: &RootDatabase, body_id: DefWithBody, module: hir::Module) -> String {
    module
        .krate(db)
        .display_name(db)
        .map(|it| it.canonical_name().as_str().to_owned())
        .into_iter()
        .chain(
            module
                .path_to_root(db)
                .into_iter()
                .filter_map(|it| it.name(db))
                .rev()
                .chain(Some(body_id.name(db).unwrap_or_else(Name::missing)))
                .map(|it| it.display(db, Edition::LATEST).to_string()),
        )
        .join("::")
}

fn location_csv_expr(db: &RootDatabase, vfs: &Vfs, sm: &BodySourceMap, expr_id: ExprId) -> String {
    let src = match sm.expr_syntax(expr_id) {
        Ok(s) => s,
        Err(SyntheticSyntax) => return "synthetic,,".to_owned(),
    };
    let root = db.parse_or_expand(src.file_id);
    let node = src.map(|e| e.to_node(&root).syntax().clone());
    let original_range = node.as_ref().original_file_range_rooted(db);
    let path = vfs.file_path(original_range.file_id.file_id(db));
    let line_index = db.line_index(original_range.file_id.file_id(db));
    let text_range = original_range.range;
    let (start, end) =
        (line_index.line_col(text_range.start()), line_index.line_col(text_range.end()));
    format!("{path},{}:{},{}:{}", start.line + 1, start.col, end.line + 1, end.col)
}

fn location_csv_pat(db: &RootDatabase, vfs: &Vfs, sm: &BodySourceMap, pat_id: PatId) -> String {
    let src = match sm.pat_syntax(pat_id) {
        Ok(s) => s,
        Err(SyntheticSyntax) => return "synthetic,,".to_owned(),
    };
    let root = db.parse_or_expand(src.file_id);
    let node = src.map(|e| e.to_node(&root).syntax().clone());
    let original_range = node.as_ref().original_file_range_rooted(db);
    let path = vfs.file_path(original_range.file_id.file_id(db));
    let line_index = db.line_index(original_range.file_id.file_id(db));
    let text_range = original_range.range;
    let (start, end) =
        (line_index.line_col(text_range.start()), line_index.line_col(text_range.end()));
    format!("{path},{}:{},{}:{}", start.line + 1, start.col, end.line + 1, end.col)
}

fn expr_syntax_range<'a>(
    db: &RootDatabase,
    vfs: &'a Vfs,
    sm: &BodySourceMap,
    expr_id: ExprId,
) -> Option<(&'a VfsPath, LineCol, LineCol)> {
    let src = sm.expr_syntax(expr_id);
    if let Ok(src) = src {
        let root = db.parse_or_expand(src.file_id);
        let node = src.map(|e| e.to_node(&root).syntax().clone());
        let original_range = node.as_ref().original_file_range_rooted(db);
        let path = vfs.file_path(original_range.file_id.file_id(db));
        let line_index = db.line_index(original_range.file_id.file_id(db));
        let text_range = original_range.range;
        let (start, end) =
            (line_index.line_col(text_range.start()), line_index.line_col(text_range.end()));
        Some((path, start, end))
    } else {
        None
    }
}
fn pat_syntax_range<'a>(
    db: &RootDatabase,
    vfs: &'a Vfs,
    sm: &BodySourceMap,
    pat_id: PatId,
) -> Option<(&'a VfsPath, LineCol, LineCol)> {
    let src = sm.pat_syntax(pat_id);
    if let Ok(src) = src {
        let root = db.parse_or_expand(src.file_id);
        let node = src.map(|e| e.to_node(&root).syntax().clone());
        let original_range = node.as_ref().original_file_range_rooted(db);
        let path = vfs.file_path(original_range.file_id.file_id(db));
        let line_index = db.line_index(original_range.file_id.file_id(db));
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

#[derive(Default, Debug, Eq, PartialEq)]
struct UsizeWithUnderscore(usize);

impl fmt::Display for UsizeWithUnderscore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num_str = self.0.to_string();

        if num_str.len() <= 3 {
            return write!(f, "{num_str}");
        }

        let mut result = String::new();

        for (count, ch) in num_str.chars().rev().enumerate() {
            if count > 0 && count % 3 == 0 {
                result.push('_');
            }
            result.push(ch);
        }

        let result = result.chars().rev().collect::<String>();
        write!(f, "{result}")
    }
}

impl std::ops::AddAssign for UsizeWithUnderscore {
    fn add_assign(&mut self, other: UsizeWithUnderscore) {
        self.0 += other.0;
    }
}

#[derive(Default, Debug, Eq, PartialEq)]
struct PrettyItemStats {
    traits: UsizeWithUnderscore,
    impls: UsizeWithUnderscore,
    mods: UsizeWithUnderscore,
    macro_calls: UsizeWithUnderscore,
    macro_rules: UsizeWithUnderscore,
}

impl From<hir_def::item_tree::ItemTreeDataStats> for PrettyItemStats {
    fn from(value: hir_def::item_tree::ItemTreeDataStats) -> Self {
        Self {
            traits: UsizeWithUnderscore(value.traits),
            impls: UsizeWithUnderscore(value.impls),
            mods: UsizeWithUnderscore(value.mods),
            macro_calls: UsizeWithUnderscore(value.macro_calls),
            macro_rules: UsizeWithUnderscore(value.macro_rules),
        }
    }
}

impl AddAssign for PrettyItemStats {
    fn add_assign(&mut self, rhs: Self) {
        self.traits += rhs.traits;
        self.impls += rhs.impls;
        self.mods += rhs.mods;
        self.macro_calls += rhs.macro_calls;
        self.macro_rules += rhs.macro_rules;
    }
}

impl fmt::Display for PrettyItemStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "traits: {}, impl: {}, mods: {}, macro calls: {}, macro rules: {}",
            self.traits, self.impls, self.mods, self.macro_calls, self.macro_rules
        )
    }
}

// FIXME(salsa-transition): bring this back whenever we implement
// Salsa's memory usage tracking to work with tracked functions.
// fn syntax_len(node: SyntaxNode) -> usize {
//     // Macro expanded code doesn't contain whitespace, so erase *all* whitespace
//     // to make macro and non-macro code comparable.
//     node.to_string().replace(|it: char| it.is_ascii_whitespace(), "").len()
// }
