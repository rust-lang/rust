#![allow(rustc::bad_opt_access)]
use crate::interface::parse_cfgspecs;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::profiling::TimePassesFormat;
use rustc_errors::{emitter::HumanReadableErrorType, registry, ColorConfig};
use rustc_session::config::rustc_optgroups;
use rustc_session::config::DebugInfo;
use rustc_session::config::Input;
use rustc_session::config::InstrumentXRay;
use rustc_session::config::LinkSelfContained;
use rustc_session::config::TraitSolver;
use rustc_session::config::{build_configuration, build_session_options, to_crate_config};
use rustc_session::config::{
    BranchProtection, Externs, OomStrategy, OutFileName, OutputType, OutputTypes, PAuthKey, PacRet,
    ProcMacroExecutionStrategy, SymbolManglingVersion, WasiExecModel,
};
use rustc_session::config::{CFGuard, ExternEntry, LinkerPluginLto, LtoCli, SwitchWithOptPath};
use rustc_session::config::{DumpMonoStatsFormat, MirSpanview};
use rustc_session::config::{ErrorOutputType, ExternLocation, LocationDetail, Options, Strip};
use rustc_session::config::{InstrumentCoverage, Passes};
use rustc_session::lint::Level;
use rustc_session::search_paths::SearchPath;
use rustc_session::utils::{CanonicalizedPath, NativeLib, NativeLibKind};
use rustc_session::{build_session, getopts, Session};
use rustc_session::{CompilerIO, EarlyErrorHandler};
use rustc_span::edition::{Edition, DEFAULT_EDITION};
use rustc_span::symbol::sym;
use rustc_span::FileName;
use rustc_span::SourceFileHashAlgorithm;
use rustc_target::spec::{CodeModel, LinkerFlavorCli, MergeFunctions, PanicStrategy, RelocModel};
use rustc_target::spec::{RelroLevel, SanitizerSet, SplitDebuginfo, StackProtector, TlsModel};

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

type CfgSpecs = FxHashSet<(String, Option<String>)>;

fn build_session_options_and_crate_config(
    handler: &mut EarlyErrorHandler,
    matches: getopts::Matches,
) -> (Options, CfgSpecs) {
    let sessopts = build_session_options(handler, &matches);
    let cfg = parse_cfgspecs(handler, matches.opt_strs("cfg"));
    (sessopts, cfg)
}

fn mk_session(handler: &mut EarlyErrorHandler, matches: getopts::Matches) -> (Session, CfgSpecs) {
    let registry = registry::Registry::new(&[]);
    let (sessopts, cfg) = build_session_options_and_crate_config(handler, matches);
    let temps_dir = sessopts.unstable_opts.temps_dir.as_deref().map(PathBuf::from);
    let io = CompilerIO {
        input: Input::Str { name: FileName::Custom(String::new()), input: String::new() },
        output_dir: None,
        output_file: None,
        temps_dir,
    };
    let sess = build_session(
        handler,
        sessopts,
        io,
        None,
        registry,
        vec![],
        Default::default(),
        None,
        None,
        "",
    );
    (sess, cfg)
}

fn new_public_extern_entry<S, I>(locations: I) -> ExternEntry
where
    S: Into<String>,
    I: IntoIterator<Item = S>,
{
    let locations: BTreeSet<CanonicalizedPath> =
        locations.into_iter().map(|s| CanonicalizedPath::new(Path::new(&s.into()))).collect();

    ExternEntry {
        location: ExternLocation::ExactPaths(locations),
        is_private_dep: false,
        add_prelude: true,
        nounused_dep: false,
        force: false,
    }
}

fn optgroups() -> getopts::Options {
    let mut opts = getopts::Options::new();
    for group in rustc_optgroups() {
        (group.apply)(&mut opts);
    }
    return opts;
}

fn mk_map<K: Ord, V>(entries: Vec<(K, V)>) -> BTreeMap<K, V> {
    BTreeMap::from_iter(entries.into_iter())
}

fn assert_same_clone(x: &Options) {
    assert_eq!(x.dep_tracking_hash(true), x.clone().dep_tracking_hash(true));
    assert_eq!(x.dep_tracking_hash(false), x.clone().dep_tracking_hash(false));
}

fn assert_same_hash(x: &Options, y: &Options) {
    assert_eq!(x.dep_tracking_hash(true), y.dep_tracking_hash(true));
    assert_eq!(x.dep_tracking_hash(false), y.dep_tracking_hash(false));
    // Check clone
    assert_same_clone(x);
    assert_same_clone(y);
}

fn assert_different_hash(x: &Options, y: &Options) {
    assert_ne!(x.dep_tracking_hash(true), y.dep_tracking_hash(true));
    assert_ne!(x.dep_tracking_hash(false), y.dep_tracking_hash(false));
    // Check clone
    assert_same_clone(x);
    assert_same_clone(y);
}

fn assert_non_crate_hash_different(x: &Options, y: &Options) {
    assert_eq!(x.dep_tracking_hash(true), y.dep_tracking_hash(true));
    assert_ne!(x.dep_tracking_hash(false), y.dep_tracking_hash(false));
    // Check clone
    assert_same_clone(x);
    assert_same_clone(y);
}

// When the user supplies --test we should implicitly supply --cfg test
#[test]
fn test_switch_implies_cfg_test() {
    rustc_span::create_default_session_globals_then(|| {
        let matches = optgroups().parse(&["--test".to_string()]).unwrap();
        let mut handler = EarlyErrorHandler::new(ErrorOutputType::default());
        let (sess, cfg) = mk_session(&mut handler, matches);
        let cfg = build_configuration(&sess, to_crate_config(cfg));
        assert!(cfg.contains(&(sym::test, None)));
    });
}

// When the user supplies --test and --cfg test, don't implicitly add another --cfg test
#[test]
fn test_switch_implies_cfg_test_unless_cfg_test() {
    rustc_span::create_default_session_globals_then(|| {
        let matches = optgroups().parse(&["--test".to_string(), "--cfg=test".to_string()]).unwrap();
        let mut handler = EarlyErrorHandler::new(ErrorOutputType::default());
        let (sess, cfg) = mk_session(&mut handler, matches);
        let cfg = build_configuration(&sess, to_crate_config(cfg));
        let mut test_items = cfg.iter().filter(|&&(name, _)| name == sym::test);
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    });
}

#[test]
fn test_can_print_warnings() {
    rustc_span::create_default_session_globals_then(|| {
        let matches = optgroups().parse(&["-Awarnings".to_string()]).unwrap();
        let mut handler = EarlyErrorHandler::new(ErrorOutputType::default());
        let (sess, _) = mk_session(&mut handler, matches);
        assert!(!sess.diagnostic().can_emit_warnings());
    });

    rustc_span::create_default_session_globals_then(|| {
        let matches =
            optgroups().parse(&["-Awarnings".to_string(), "-Dwarnings".to_string()]).unwrap();
        let mut handler = EarlyErrorHandler::new(ErrorOutputType::default());
        let (sess, _) = mk_session(&mut handler, matches);
        assert!(sess.diagnostic().can_emit_warnings());
    });

    rustc_span::create_default_session_globals_then(|| {
        let matches = optgroups().parse(&["-Adead_code".to_string()]).unwrap();
        let mut handler = EarlyErrorHandler::new(ErrorOutputType::default());
        let (sess, _) = mk_session(&mut handler, matches);
        assert!(sess.diagnostic().can_emit_warnings());
    });
}

#[test]
fn test_output_types_tracking_hash_different_paths() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();

    v1.output_types = OutputTypes::new(&[(
        OutputType::Exe,
        Some(OutFileName::Real(PathBuf::from("./some/thing"))),
    )]);
    v2.output_types = OutputTypes::new(&[(
        OutputType::Exe,
        Some(OutFileName::Real(PathBuf::from("/some/thing"))),
    )]);
    v3.output_types = OutputTypes::new(&[(OutputType::Exe, None)]);

    assert_non_crate_hash_different(&v1, &v2);
    assert_non_crate_hash_different(&v1, &v3);
    assert_non_crate_hash_different(&v2, &v3);
}

#[test]
fn test_output_types_tracking_hash_different_construction_order() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();

    v1.output_types = OutputTypes::new(&[
        (OutputType::Exe, Some(OutFileName::Real(PathBuf::from("./some/thing")))),
        (OutputType::Bitcode, Some(OutFileName::Real(PathBuf::from("./some/thing.bc")))),
    ]);

    v2.output_types = OutputTypes::new(&[
        (OutputType::Bitcode, Some(OutFileName::Real(PathBuf::from("./some/thing.bc")))),
        (OutputType::Exe, Some(OutFileName::Real(PathBuf::from("./some/thing")))),
    ]);

    assert_same_hash(&v1, &v2);
}

#[test]
fn test_externs_tracking_hash_different_construction_order() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();

    v1.externs = Externs::new(mk_map(vec![
        (String::from("a"), new_public_extern_entry(vec!["b", "c"])),
        (String::from("d"), new_public_extern_entry(vec!["e", "f"])),
    ]));

    v2.externs = Externs::new(mk_map(vec![
        (String::from("d"), new_public_extern_entry(vec!["e", "f"])),
        (String::from("a"), new_public_extern_entry(vec!["b", "c"])),
    ]));

    v3.externs = Externs::new(mk_map(vec![
        (String::from("a"), new_public_extern_entry(vec!["b", "c"])),
        (String::from("d"), new_public_extern_entry(vec!["f", "e"])),
    ]));

    assert_same_hash(&v1, &v2);
    assert_same_hash(&v1, &v3);
    assert_same_hash(&v2, &v3);
}

#[test]
fn test_lints_tracking_hash_different_values() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();

    v1.lint_opts = vec![
        (String::from("a"), Level::Allow),
        (String::from("b"), Level::Warn),
        (String::from("c"), Level::Deny),
        (String::from("d"), Level::Forbid),
    ];

    v2.lint_opts = vec![
        (String::from("a"), Level::Allow),
        (String::from("b"), Level::Warn),
        (String::from("X"), Level::Deny),
        (String::from("d"), Level::Forbid),
    ];

    v3.lint_opts = vec![
        (String::from("a"), Level::Allow),
        (String::from("b"), Level::Warn),
        (String::from("c"), Level::Forbid),
        (String::from("d"), Level::Deny),
    ];

    assert_non_crate_hash_different(&v1, &v2);
    assert_non_crate_hash_different(&v1, &v3);
    assert_non_crate_hash_different(&v2, &v3);
}

#[test]
fn test_lints_tracking_hash_different_construction_order() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();

    v1.lint_opts = vec![
        (String::from("a"), Level::Allow),
        (String::from("b"), Level::Warn),
        (String::from("c"), Level::Deny),
        (String::from("d"), Level::Forbid),
    ];

    v2.lint_opts = vec![
        (String::from("a"), Level::Allow),
        (String::from("c"), Level::Deny),
        (String::from("b"), Level::Warn),
        (String::from("d"), Level::Forbid),
    ];

    // The hash should be order-dependent
    assert_non_crate_hash_different(&v1, &v2);
}

#[test]
fn test_lint_cap_hash_different() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let v3 = Options::default();

    v1.lint_cap = Some(Level::Forbid);
    v2.lint_cap = Some(Level::Allow);

    assert_non_crate_hash_different(&v1, &v2);
    assert_non_crate_hash_different(&v1, &v3);
    assert_non_crate_hash_different(&v2, &v3);
}

#[test]
fn test_search_paths_tracking_hash_different_order() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();
    let mut v4 = Options::default();

    let handler = EarlyErrorHandler::new(JSON);
    const JSON: ErrorOutputType = ErrorOutputType::Json {
        pretty: false,
        json_rendered: HumanReadableErrorType::Default(ColorConfig::Never),
    };

    // Reference
    v1.search_paths.push(SearchPath::from_cli_opt(&handler, "native=abc"));
    v1.search_paths.push(SearchPath::from_cli_opt(&handler, "crate=def"));
    v1.search_paths.push(SearchPath::from_cli_opt(&handler, "dependency=ghi"));
    v1.search_paths.push(SearchPath::from_cli_opt(&handler, "framework=jkl"));
    v1.search_paths.push(SearchPath::from_cli_opt(&handler, "all=mno"));

    v2.search_paths.push(SearchPath::from_cli_opt(&handler, "native=abc"));
    v2.search_paths.push(SearchPath::from_cli_opt(&handler, "dependency=ghi"));
    v2.search_paths.push(SearchPath::from_cli_opt(&handler, "crate=def"));
    v2.search_paths.push(SearchPath::from_cli_opt(&handler, "framework=jkl"));
    v2.search_paths.push(SearchPath::from_cli_opt(&handler, "all=mno"));

    v3.search_paths.push(SearchPath::from_cli_opt(&handler, "crate=def"));
    v3.search_paths.push(SearchPath::from_cli_opt(&handler, "framework=jkl"));
    v3.search_paths.push(SearchPath::from_cli_opt(&handler, "native=abc"));
    v3.search_paths.push(SearchPath::from_cli_opt(&handler, "dependency=ghi"));
    v3.search_paths.push(SearchPath::from_cli_opt(&handler, "all=mno"));

    v4.search_paths.push(SearchPath::from_cli_opt(&handler, "all=mno"));
    v4.search_paths.push(SearchPath::from_cli_opt(&handler, "native=abc"));
    v4.search_paths.push(SearchPath::from_cli_opt(&handler, "crate=def"));
    v4.search_paths.push(SearchPath::from_cli_opt(&handler, "dependency=ghi"));
    v4.search_paths.push(SearchPath::from_cli_opt(&handler, "framework=jkl"));

    assert_same_hash(&v1, &v2);
    assert_same_hash(&v1, &v3);
    assert_same_hash(&v1, &v4);
}

#[test]
fn test_native_libs_tracking_hash_different_values() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();
    let mut v4 = Options::default();
    let mut v5 = Options::default();

    // Reference
    v1.libs = vec![
        NativeLib {
            name: String::from("a"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("b"),
            new_name: None,
            kind: NativeLibKind::Framework { as_needed: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("c"),
            new_name: None,
            kind: NativeLibKind::Unspecified,
            verbatim: None,
        },
    ];

    // Change label
    v2.libs = vec![
        NativeLib {
            name: String::from("a"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("X"),
            new_name: None,
            kind: NativeLibKind::Framework { as_needed: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("c"),
            new_name: None,
            kind: NativeLibKind::Unspecified,
            verbatim: None,
        },
    ];

    // Change kind
    v3.libs = vec![
        NativeLib {
            name: String::from("a"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("b"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("c"),
            new_name: None,
            kind: NativeLibKind::Unspecified,
            verbatim: None,
        },
    ];

    // Change new-name
    v4.libs = vec![
        NativeLib {
            name: String::from("a"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("b"),
            new_name: Some(String::from("X")),
            kind: NativeLibKind::Framework { as_needed: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("c"),
            new_name: None,
            kind: NativeLibKind::Unspecified,
            verbatim: None,
        },
    ];

    // Change verbatim
    v5.libs = vec![
        NativeLib {
            name: String::from("a"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("b"),
            new_name: None,
            kind: NativeLibKind::Framework { as_needed: None },
            verbatim: Some(true),
        },
        NativeLib {
            name: String::from("c"),
            new_name: None,
            kind: NativeLibKind::Unspecified,
            verbatim: None,
        },
    ];

    assert_different_hash(&v1, &v2);
    assert_different_hash(&v1, &v3);
    assert_different_hash(&v1, &v4);
    assert_different_hash(&v1, &v5);
}

#[test]
fn test_native_libs_tracking_hash_different_order() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();

    // Reference
    v1.libs = vec![
        NativeLib {
            name: String::from("a"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("b"),
            new_name: None,
            kind: NativeLibKind::Framework { as_needed: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("c"),
            new_name: None,
            kind: NativeLibKind::Unspecified,
            verbatim: None,
        },
    ];

    v2.libs = vec![
        NativeLib {
            name: String::from("b"),
            new_name: None,
            kind: NativeLibKind::Framework { as_needed: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("a"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("c"),
            new_name: None,
            kind: NativeLibKind::Unspecified,
            verbatim: None,
        },
    ];

    v3.libs = vec![
        NativeLib {
            name: String::from("c"),
            new_name: None,
            kind: NativeLibKind::Unspecified,
            verbatim: None,
        },
        NativeLib {
            name: String::from("a"),
            new_name: None,
            kind: NativeLibKind::Static { bundle: None, whole_archive: None },
            verbatim: None,
        },
        NativeLib {
            name: String::from("b"),
            new_name: None,
            kind: NativeLibKind::Framework { as_needed: None },
            verbatim: None,
        },
    ];

    // The hash should be order-dependent
    assert_different_hash(&v1, &v2);
    assert_different_hash(&v1, &v3);
    assert_different_hash(&v2, &v3);
}

#[test]
fn test_codegen_options_tracking_hash() {
    let reference = Options::default();
    let mut opts = Options::default();

    macro_rules! untracked {
        ($name: ident, $non_default_value: expr) => {
            assert_ne!(opts.cg.$name, $non_default_value);
            opts.cg.$name = $non_default_value;
            assert_same_hash(&reference, &opts);
        };
    }

    // Make sure that changing an [UNTRACKED] option leaves the hash unchanged.
    // tidy-alphabetical-start
    untracked!(ar, String::from("abc"));
    untracked!(codegen_units, Some(42));
    untracked!(default_linker_libraries, true);
    untracked!(dlltool, Some(PathBuf::from("custom_dlltool.exe")));
    untracked!(extra_filename, String::from("extra-filename"));
    untracked!(incremental, Some(String::from("abc")));
    // `link_arg` is omitted because it just forwards to `link_args`.
    untracked!(link_args, vec![String::from("abc"), String::from("def")]);
    untracked!(link_self_contained, LinkSelfContained::on());
    untracked!(linker, Some(PathBuf::from("linker")));
    untracked!(linker_flavor, Some(LinkerFlavorCli::Gcc));
    untracked!(no_stack_check, true);
    untracked!(remark, Passes::Some(vec![String::from("pass1"), String::from("pass2")]));
    untracked!(rpath, true);
    untracked!(save_temps, true);
    untracked!(strip, Strip::Debuginfo);
    // tidy-alphabetical-end

    macro_rules! tracked {
        ($name: ident, $non_default_value: expr) => {
            opts = reference.clone();
            assert_ne!(opts.cg.$name, $non_default_value);
            opts.cg.$name = $non_default_value;
            assert_different_hash(&reference, &opts);
        };
    }

    // Make sure that changing a [TRACKED] option changes the hash.
    // tidy-alphabetical-start
    tracked!(code_model, Some(CodeModel::Large));
    tracked!(control_flow_guard, CFGuard::Checks);
    tracked!(debug_assertions, Some(true));
    tracked!(debuginfo, DebugInfo::Limited);
    tracked!(embed_bitcode, false);
    tracked!(force_frame_pointers, Some(false));
    tracked!(force_unwind_tables, Some(true));
    tracked!(inline_threshold, Some(0xf007ba11));
    tracked!(instrument_coverage, Some(InstrumentCoverage::All));
    tracked!(link_dead_code, Some(true));
    tracked!(linker_plugin_lto, LinkerPluginLto::LinkerPluginAuto);
    tracked!(llvm_args, vec![String::from("1"), String::from("2")]);
    tracked!(lto, LtoCli::Fat);
    tracked!(metadata, vec![String::from("A"), String::from("B")]);
    tracked!(no_prepopulate_passes, true);
    tracked!(no_redzone, Some(true));
    tracked!(no_vectorize_loops, true);
    tracked!(no_vectorize_slp, true);
    tracked!(opt_level, "3".to_string());
    tracked!(overflow_checks, Some(true));
    tracked!(panic, Some(PanicStrategy::Abort));
    tracked!(passes, vec![String::from("1"), String::from("2")]);
    tracked!(prefer_dynamic, true);
    tracked!(profile_generate, SwitchWithOptPath::Enabled(None));
    tracked!(profile_use, Some(PathBuf::from("abc")));
    tracked!(relocation_model, Some(RelocModel::Pic));
    tracked!(soft_float, true);
    tracked!(split_debuginfo, Some(SplitDebuginfo::Packed));
    tracked!(symbol_mangling_version, Some(SymbolManglingVersion::V0));
    tracked!(target_cpu, Some(String::from("abc")));
    tracked!(target_feature, String::from("all the features, all of them"));
    // tidy-alphabetical-end
}

#[test]
fn test_top_level_options_tracked_no_crate() {
    let reference = Options::default();
    let mut opts;

    macro_rules! tracked {
        ($name: ident, $non_default_value: expr) => {
            opts = reference.clone();
            assert_ne!(opts.$name, $non_default_value);
            opts.$name = $non_default_value;
            // The crate hash should be the same
            assert_eq!(reference.dep_tracking_hash(true), opts.dep_tracking_hash(true));
            // The incremental hash should be different
            assert_ne!(reference.dep_tracking_hash(false), opts.dep_tracking_hash(false));
        };
    }

    // Make sure that changing a [TRACKED_NO_CRATE_HASH] option leaves the crate hash unchanged but changes the incremental hash.
    // tidy-alphabetical-start
    tracked!(
        real_rust_source_base_dir,
        Some("/home/bors/rust/.rustup/toolchains/nightly/lib/rustlib/src/rust".into())
    );
    tracked!(remap_path_prefix, vec![("/home/bors/rust".into(), "src".into())]);
    // tidy-alphabetical-end
}

#[test]
fn test_unstable_options_tracking_hash() {
    let reference = Options::default();
    let mut opts = Options::default();

    macro_rules! untracked {
        ($name: ident, $non_default_value: expr) => {
            assert_ne!(opts.unstable_opts.$name, $non_default_value);
            opts.unstable_opts.$name = $non_default_value;
            assert_same_hash(&reference, &opts);
        };
    }

    // Make sure that changing an [UNTRACKED] option leaves the hash unchanged.
    // tidy-alphabetical-start
    untracked!(assert_incr_state, Some(String::from("loaded")));
    untracked!(deduplicate_diagnostics, false);
    untracked!(dep_tasks, true);
    untracked!(dont_buffer_diagnostics, true);
    untracked!(dump_dep_graph, true);
    untracked!(dump_drop_tracking_cfg, Some("cfg.dot".to_string()));
    untracked!(dump_mir, Some(String::from("abc")));
    untracked!(dump_mir_dataflow, true);
    untracked!(dump_mir_dir, String::from("abc"));
    untracked!(dump_mir_exclude_pass_number, true);
    untracked!(dump_mir_graphviz, true);
    untracked!(dump_mir_spanview, Some(MirSpanview::Statement));
    untracked!(dump_mono_stats, SwitchWithOptPath::Enabled(Some("mono-items-dir/".into())));
    untracked!(dump_mono_stats_format, DumpMonoStatsFormat::Json);
    untracked!(dylib_lto, true);
    untracked!(emit_stack_sizes, true);
    untracked!(future_incompat_test, true);
    untracked!(hir_stats, true);
    untracked!(identify_regions, true);
    untracked!(incremental_info, true);
    untracked!(incremental_verify_ich, true);
    untracked!(input_stats, true);
    untracked!(keep_hygiene_data, true);
    untracked!(link_native_libraries, false);
    untracked!(llvm_time_trace, true);
    untracked!(ls, true);
    untracked!(macro_backtrace, true);
    untracked!(meta_stats, true);
    untracked!(mir_include_spans, true);
    untracked!(nll_facts, true);
    untracked!(no_analysis, true);
    untracked!(no_leak_check, true);
    untracked!(no_parallel_llvm, true);
    untracked!(parse_only, true);
    untracked!(perf_stats, true);
    // `pre_link_arg` is omitted because it just forwards to `pre_link_args`.
    untracked!(pre_link_args, vec![String::from("abc"), String::from("def")]);
    untracked!(print_llvm_passes, true);
    untracked!(print_mono_items, Some(String::from("abc")));
    untracked!(print_type_sizes, true);
    untracked!(proc_macro_backtrace, true);
    untracked!(proc_macro_execution_strategy, ProcMacroExecutionStrategy::CrossThread);
    untracked!(profile_closures, true);
    untracked!(query_dep_graph, true);
    untracked!(self_profile, SwitchWithOptPath::Enabled(None));
    untracked!(self_profile_events, Some(vec![String::new()]));
    untracked!(span_debug, true);
    untracked!(span_free_formats, true);
    untracked!(temps_dir, Some(String::from("abc")));
    untracked!(threads, 99);
    untracked!(time_llvm_passes, true);
    untracked!(time_passes, true);
    untracked!(time_passes_format, TimePassesFormat::Json);
    untracked!(trace_macros, true);
    untracked!(track_diagnostics, true);
    untracked!(trim_diagnostic_paths, false);
    untracked!(ui_testing, true);
    untracked!(unpretty, Some("expanded".to_string()));
    untracked!(unstable_options, true);
    untracked!(validate_mir, true);
    untracked!(verbose, true);
    // tidy-alphabetical-end

    macro_rules! tracked {
        ($name: ident, $non_default_value: expr) => {
            opts = reference.clone();
            assert_ne!(opts.unstable_opts.$name, $non_default_value);
            opts.unstable_opts.$name = $non_default_value;
            assert_different_hash(&reference, &opts);
        };
    }

    // Make sure that changing a [TRACKED] option changes the hash.
    // tidy-alphabetical-start
    tracked!(allow_features, Some(vec![String::from("lang_items")]));
    tracked!(always_encode_mir, true);
    tracked!(asm_comments, true);
    tracked!(assume_incomplete_release, true);
    tracked!(binary_dep_depinfo, true);
    tracked!(box_noalias, false);
    tracked!(
        branch_protection,
        Some(BranchProtection {
            bti: true,
            pac_ret: Some(PacRet { leaf: true, key: PAuthKey::B })
        })
    );
    tracked!(codegen_backend, Some("abc".to_string()));
    tracked!(crate_attr, vec!["abc".to_string()]);
    tracked!(debug_info_for_profiling, true);
    tracked!(debug_macros, true);
    tracked!(dep_info_omit_d_target, true);
    tracked!(drop_tracking, true);
    tracked!(dual_proc_macros, true);
    tracked!(dwarf_version, Some(5));
    tracked!(emit_thin_lto, false);
    tracked!(export_executable_symbols, true);
    tracked!(fewer_names, Some(true));
    tracked!(flatten_format_args, false);
    tracked!(force_unstable_if_unmarked, true);
    tracked!(fuel, Some(("abc".to_string(), 99)));
    tracked!(function_sections, Some(false));
    tracked!(human_readable_cgu_names, true);
    tracked!(incremental_ignore_spans, true);
    tracked!(inline_in_all_cgus, Some(true));
    tracked!(inline_mir, Some(true));
    tracked!(inline_mir_hint_threshold, Some(123));
    tracked!(inline_mir_threshold, Some(123));
    tracked!(instrument_coverage, Some(InstrumentCoverage::All));
    tracked!(instrument_mcount, true);
    tracked!(instrument_xray, Some(InstrumentXRay::default()));
    tracked!(link_directives, false);
    tracked!(link_only, true);
    tracked!(llvm_plugins, vec![String::from("plugin_name")]);
    tracked!(location_detail, LocationDetail { file: true, line: false, column: false });
    tracked!(maximal_hir_to_mir_coverage, true);
    tracked!(merge_functions, Some(MergeFunctions::Disabled));
    tracked!(mir_emit_retag, true);
    tracked!(mir_enable_passes, vec![("DestProp".to_string(), false)]);
    tracked!(mir_keep_place_mention, true);
    tracked!(mir_opt_level, Some(4));
    tracked!(move_size_limit, Some(4096));
    tracked!(mutable_noalias, false);
    tracked!(no_generate_arange_section, true);
    tracked!(no_jump_tables, true);
    tracked!(no_link, true);
    tracked!(no_profiler_runtime, true);
    tracked!(no_unique_section_names, true);
    tracked!(oom, OomStrategy::Panic);
    tracked!(osx_rpath_install_name, true);
    tracked!(packed_bundled_libs, true);
    tracked!(panic_abort_tests, true);
    tracked!(panic_in_drop, PanicStrategy::Abort);
    tracked!(plt, Some(true));
    tracked!(polonius, true);
    tracked!(precise_enum_drop_elaboration, false);
    tracked!(print_fuel, Some("abc".to_string()));
    tracked!(profile, true);
    tracked!(profile_emit, Some(PathBuf::from("abc")));
    tracked!(profile_sample_use, Some(PathBuf::from("abc")));
    tracked!(profiler_runtime, "abc".to_string());
    tracked!(relax_elf_relocations, Some(true));
    tracked!(relro_level, Some(RelroLevel::Full));
    tracked!(remap_cwd_prefix, Some(PathBuf::from("abc")));
    tracked!(report_delayed_bugs, true);
    tracked!(sanitizer, SanitizerSet::ADDRESS);
    tracked!(sanitizer_cfi_canonical_jump_tables, None);
    tracked!(sanitizer_cfi_generalize_pointers, Some(true));
    tracked!(sanitizer_cfi_normalize_integers, Some(true));
    tracked!(sanitizer_memory_track_origins, 2);
    tracked!(sanitizer_recover, SanitizerSet::ADDRESS);
    tracked!(saturating_float_casts, Some(true));
    tracked!(share_generics, Some(true));
    tracked!(show_span, Some(String::from("abc")));
    tracked!(simulate_remapped_rust_src_base, Some(PathBuf::from("/rustc/abc")));
    tracked!(split_lto_unit, Some(true));
    tracked!(src_hash_algorithm, Some(SourceFileHashAlgorithm::Sha1));
    tracked!(stack_protector, StackProtector::All);
    tracked!(symbol_mangling_version, Some(SymbolManglingVersion::V0));
    tracked!(teach, true);
    tracked!(thinlto, Some(true));
    tracked!(thir_unsafeck, true);
    tracked!(tiny_const_eval_limit, true);
    tracked!(tls_model, Some(TlsModel::GeneralDynamic));
    tracked!(trait_solver, TraitSolver::NextCoherence);
    tracked!(translate_remapped_path_to_local_path, false);
    tracked!(trap_unreachable, Some(false));
    tracked!(treat_err_as_bug, NonZeroUsize::new(1));
    tracked!(tune_cpu, Some(String::from("abc")));
    tracked!(uninit_const_chunk_threshold, 123);
    tracked!(unleash_the_miri_inside_of_you, true);
    tracked!(use_ctors_section, Some(true));
    tracked!(verify_llvm_ir, true);
    tracked!(virtual_function_elimination, true);
    tracked!(wasi_exec_model, Some(WasiExecModel::Reactor));
    // tidy-alphabetical-end

    macro_rules! tracked_no_crate_hash {
        ($name: ident, $non_default_value: expr) => {
            opts = reference.clone();
            assert_ne!(opts.unstable_opts.$name, $non_default_value);
            opts.unstable_opts.$name = $non_default_value;
            assert_non_crate_hash_different(&reference, &opts);
        };
    }
    tracked_no_crate_hash!(no_codegen, true);
}

#[test]
fn test_edition_parsing() {
    // test default edition
    let options = Options::default();
    assert!(options.edition == DEFAULT_EDITION);

    let mut handler = EarlyErrorHandler::new(ErrorOutputType::default());

    let matches = optgroups().parse(&["--edition=2018".to_string()]).unwrap();
    let (sessopts, _) = build_session_options_and_crate_config(&mut handler, matches);
    assert!(sessopts.edition == Edition::Edition2018)
}
