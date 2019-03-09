use util;
use proc_macro_decls;

use rustc::dep_graph::DepGraph;
use rustc::hir;
use rustc::hir::lowering::lower_crate;
use rustc::hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc::lint;
use rustc::middle::{self, reachable, resolve_lifetime, stability};
use rustc::middle::privacy::AccessLevels;
use rustc::ty::{self, AllArenas, Resolutions, TyCtxt};
use rustc::ty::steal::Steal;
use rustc::traits;
use rustc::util::common::{time, ErrorReported};
use rustc::util::profiling::ProfileCategory;
use rustc::session::{CompileResult, CrateDisambiguator, Session};
use rustc::session::config::{self, Input, OutputFilenames, OutputType};
use rustc::session::search_paths::PathKind;
use rustc_allocator as allocator;
use rustc_borrowck as borrowck;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{Lrc, ParallelIterator, par_iter};
use rustc_incremental;
use rustc_metadata::creader::CrateLoader;
use rustc_metadata::cstore::{self, CStore};
use rustc_mir as mir;
use rustc_passes::{self, ast_validation, hir_stats, loops, rvalue_promotion, layout_test};
use rustc_plugin as plugin;
use rustc_plugin::registry::Registry;
use rustc_privacy;
use rustc_resolve::{Resolver, ResolverArenas};
use rustc_traits;
use rustc_typeck as typeck;
use syntax::{self, ast, attr, diagnostics, visit};
use syntax::early_buffered_lints::BufferedEarlyLint;
use syntax::ext::base::ExtCtxt;
use syntax::mut_visit::MutVisitor;
use syntax::parse::{self, PResult};
use syntax::util::node_count::NodeCounter;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax::symbol::Symbol;
use syntax_pos::{FileName, hygiene};
use syntax_ext;

use serialize::json;

use std::any::Any;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::iter;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::cell::RefCell;
use std::rc::Rc;
use std::mem;
use std::ops::Generator;

/// Returns all the paths that correspond to generated files.
pub fn generated_output_paths(
    sess: &Session,
    outputs: &OutputFilenames,
    exact_name: bool,
    crate_name: &str,
) -> Vec<PathBuf> {
    let mut out_filenames = Vec::new();
    for output_type in sess.opts.output_types.keys() {
        let file = outputs.path(*output_type);
        match *output_type {
            // If the filename has been overridden using `-o`, it will not be modified
            // by appending `.rlib`, `.exe`, etc., so we can skip this transformation.
            OutputType::Exe if !exact_name => for crate_type in sess.crate_types.borrow().iter() {
                let p = ::rustc_codegen_utils::link::filename_for_input(
                    sess,
                    *crate_type,
                    crate_name,
                    outputs,
                );
                out_filenames.push(p);
            },
            OutputType::DepInfo if sess.opts.debugging_opts.dep_info_omit_d_target => {
                // Don't add the dep-info output when omitting it from dep-info targets
            }
            _ => {
                out_filenames.push(file);
            }
        }
    }
    out_filenames
}

// Runs `f` on every output file path and returns the first non-None result, or None if `f`
// returns None for every file path.
fn check_output<F, T>(output_paths: &[PathBuf], f: F) -> Option<T>
where
    F: Fn(&PathBuf) -> Option<T>,
{
    for output_path in output_paths {
        if let Some(result) = f(output_path) {
            return Some(result);
        }
    }
    None
}

pub fn output_contains_path(output_paths: &[PathBuf], input_path: &PathBuf) -> bool {
    let input_path = input_path.canonicalize().ok();
    if input_path.is_none() {
        return false;
    }
    let check = |output_path: &PathBuf| {
        if output_path.canonicalize().ok() == input_path {
            Some(())
        } else {
            None
        }
    };
    check_output(output_paths, check).is_some()
}

pub fn output_conflicts_with_dir(output_paths: &[PathBuf]) -> Option<PathBuf> {
    let check = |output_path: &PathBuf| {
        if output_path.is_dir() {
            Some(output_path.clone())
        } else {
            None
        }
    };
    check_output(output_paths, check)
}

fn escape_dep_filename(filename: &FileName) -> String {
    // Apparently clang and gcc *only* escape spaces:
    // http://llvm.org/klaus/clang/commit/9d50634cfc268ecc9a7250226dd5ca0e945240d4
    filename.to_string().replace(" ", "\\ ")
}

pub fn write_out_deps(sess: &Session, outputs: &OutputFilenames, out_filenames: &[PathBuf]) {
    // Write out dependency rules to the dep-info file if requested
    if !sess.opts.output_types.contains_key(&OutputType::DepInfo) {
        return;
    }
    let deps_filename = outputs.path(OutputType::DepInfo);

    let result = (|| -> io::Result<()> {
        // Build a list of files used to compile the output and
        // write Makefile-compatible dependency rules
        let files: Vec<String> = sess.source_map()
            .files()
            .iter()
            .filter(|fmap| fmap.is_real_file())
            .filter(|fmap| !fmap.is_imported())
            .map(|fmap| escape_dep_filename(&fmap.name))
            .collect();
        let mut file = fs::File::create(&deps_filename)?;
        for path in out_filenames {
            writeln!(file, "{}: {}\n", path.display(), files.join(" "))?;
        }

        // Emit a fake target for each input file to the compilation. This
        // prevents `make` from spitting out an error if a file is later
        // deleted. For more info see #28735
        for path in files {
            writeln!(file, "{}:", path)?;
        }
        Ok(())
    })();

    if let Err(e) = result {
        sess.fatal(&format!(
            "error writing dependencies to `{}`: {}",
            deps_filename.display(),
            e
        ));
    }
}

pub fn provide(providers: &mut ty::query::Providers) {
    providers.analysis = analysis;
    proc_macro_decls::provide(providers);
}

fn analysis<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    cnum: CrateNum,
) -> Result<(), ErrorReported> {
    assert_eq!(cnum, LOCAL_CRATE);

    let sess = tcx.sess;

    time(sess, "misc checking 1", || {
        parallel!({
            time(sess, "looking for entry point", || {
                middle::entry::find_entry_point(tcx)
            });

            time(sess, "looking for plugin registrar", || {
                plugin::build::find_plugin_registrar(tcx)
            });

            time(sess, "looking for derive registrar", || {
                proc_macro_decls::find(tcx)
            });
        }, {
            par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                tcx.ensure().check_mod_loops(tcx.hir().local_def_id(module));
                tcx.ensure().check_mod_attrs(tcx.hir().local_def_id(module));
                tcx.ensure().check_mod_unstable_api_usage(tcx.hir().local_def_id(module));
            });
        });
    });

    // passes are timed inside typeck
    typeck::check_crate(tcx)?;

    time(sess, "misc checking 2", || {
        parallel!({
            time(sess, "rvalue promotion + match checking", || {
                tcx.par_body_owners(|def_id| {
                    tcx.ensure().const_is_rvalue_promotable_to_static(def_id);
                    tcx.ensure().check_match(def_id);
                });
            });
        }, {
            time(sess, "liveness checking + intrinsic checking", || {
                par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                    // this must run before MIR dump, because
                    // "not all control paths return a value" is reported here.
                    //
                    // maybe move the check to a MIR pass?
                    tcx.ensure().check_mod_liveness(tcx.hir().local_def_id(module));

                    tcx.ensure().check_mod_intrinsics(tcx.hir().local_def_id(module));
                });
            });
        });
    });

    // Abort so we don't try to construct MIR with liveness errors.
    // We also won't want to continue with errors from rvalue promotion
    tcx.sess.abort_if_errors();

    time(sess, "borrow checking", || {
        if tcx.use_ast_borrowck() {
            borrowck::check_crate(tcx);
        }
    });

    time(sess,
            "MIR borrow checking",
            || tcx.par_body_owners(|def_id| { tcx.ensure().mir_borrowck(def_id); }));

    time(sess, "dumping chalk-like clauses", || {
        rustc_traits::lowering::dump_program_clauses(tcx);
    });

    time(sess, "MIR effect checking", || {
        for def_id in tcx.body_owners() {
            mir::transform::check_unsafety::check_unsafety(tcx, def_id)
        }
    });

    time(sess, "layout testing", || layout_test::test_layout(tcx));

    // Avoid overwhelming user with errors if borrow checking failed.
    // I'm not sure how helpful this is, to be honest, but it avoids
    // a
    // lot of annoying errors in the compile-fail tests (basically,
    // lint warnings and so on -- kindck used to do this abort, but
    // kindck is gone now). -nmatsakis
    if sess.err_count() > 0 {
        return Err(ErrorReported);
    }

    time(sess, "misc checking 3", || {
        parallel!({
            time(sess, "privacy access levels", || {
                tcx.ensure().privacy_access_levels(LOCAL_CRATE);
            });
            parallel!({
                time(sess, "private in public", || {
                    tcx.ensure().check_private_in_public(LOCAL_CRATE);
                });
            }, {
                time(sess, "death checking", || middle::dead::check_crate(tcx));
            },  {
                time(sess, "unused lib feature checking", || {
                    stability::check_unused_or_stable_features(tcx)
                });
            }, {
                time(sess, "lint checking", || lint::check_crate(tcx));
            });
        }, {
            time(sess, "privacy checking modules", || {
                par_iter(&tcx.hir().krate().modules).for_each(|(&module, _)| {
                    tcx.ensure().check_mod_privacy(tcx.hir().local_def_id(module));
                });
            });
        });
    });

    Ok(())
}
