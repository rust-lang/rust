use crate::interface::parse_cfgspecs;

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{emitter::HumanReadableErrorType, registry, ColorConfig};
use rustc_middle::middle::cstore;
use rustc_session::config::{build_configuration, build_session_options, to_crate_config};
use rustc_session::config::{rustc_optgroups, ErrorOutputType, ExternLocation, Options, Passes};
use rustc_session::config::{CFGuard, ExternEntry, LinkerPluginLto, LtoCli, SwitchWithOptPath};
use rustc_session::config::{Externs, OutputType, OutputTypes, Sanitizer, SymbolManglingVersion};
use rustc_session::getopts;
use rustc_session::lint::Level;
use rustc_session::search_paths::SearchPath;
use rustc_session::{build_session, Session};
use rustc_span::edition::{Edition, DEFAULT_EDITION};
use rustc_span::symbol::sym;
use rustc_span::SourceFileHashAlgorithm;
use rustc_target::spec::{LinkerFlavor, MergeFunctions, PanicStrategy, RelroLevel};
use std::collections::{BTreeMap, BTreeSet};
use std::iter::FromIterator;
use std::path::PathBuf;

type CfgSpecs = FxHashSet<(String, Option<String>)>;

fn build_session_options_and_crate_config(matches: getopts::Matches) -> (Options, CfgSpecs) {
    let sessopts = build_session_options(&matches);
    let cfg = parse_cfgspecs(matches.opt_strs("cfg"));
    (sessopts, cfg)
}

fn mk_session(matches: getopts::Matches) -> (Session, CfgSpecs) {
    let registry = registry::Registry::new(&[]);
    let (sessopts, cfg) = build_session_options_and_crate_config(matches);
    let sess = build_session(sessopts, None, registry);
    (sess, cfg)
}

fn new_public_extern_entry<S, I>(locations: I) -> ExternEntry
where
    S: Into<String>,
    I: IntoIterator<Item = S>,
{
    let locations: BTreeSet<_> = locations.into_iter().map(|s| s.into()).collect();

    ExternEntry {
        location: ExternLocation::ExactPaths(locations),
        is_private_dep: false,
        add_prelude: true,
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

// When the user supplies --test we should implicitly supply --cfg test
#[test]
fn test_switch_implies_cfg_test() {
    rustc_ast::with_default_globals(|| {
        let matches = optgroups().parse(&["--test".to_string()]).unwrap();
        let (sess, cfg) = mk_session(matches);
        let cfg = build_configuration(&sess, to_crate_config(cfg));
        assert!(cfg.contains(&(sym::test, None)));
    });
}

// When the user supplies --test and --cfg test, don't implicitly add another --cfg test
#[test]
fn test_switch_implies_cfg_test_unless_cfg_test() {
    rustc_ast::with_default_globals(|| {
        let matches = optgroups().parse(&["--test".to_string(), "--cfg=test".to_string()]).unwrap();
        let (sess, cfg) = mk_session(matches);
        let cfg = build_configuration(&sess, to_crate_config(cfg));
        let mut test_items = cfg.iter().filter(|&&(name, _)| name == sym::test);
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    });
}

#[test]
fn test_can_print_warnings() {
    rustc_ast::with_default_globals(|| {
        let matches = optgroups().parse(&["-Awarnings".to_string()]).unwrap();
        let (sess, _) = mk_session(matches);
        assert!(!sess.diagnostic().can_emit_warnings());
    });

    rustc_ast::with_default_globals(|| {
        let matches =
            optgroups().parse(&["-Awarnings".to_string(), "-Dwarnings".to_string()]).unwrap();
        let (sess, _) = mk_session(matches);
        assert!(sess.diagnostic().can_emit_warnings());
    });

    rustc_ast::with_default_globals(|| {
        let matches = optgroups().parse(&["-Adead_code".to_string()]).unwrap();
        let (sess, _) = mk_session(matches);
        assert!(sess.diagnostic().can_emit_warnings());
    });
}

#[test]
fn test_output_types_tracking_hash_different_paths() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();

    v1.output_types = OutputTypes::new(&[(OutputType::Exe, Some(PathBuf::from("./some/thing")))]);
    v2.output_types = OutputTypes::new(&[(OutputType::Exe, Some(PathBuf::from("/some/thing")))]);
    v3.output_types = OutputTypes::new(&[(OutputType::Exe, None)]);

    assert!(v1.dep_tracking_hash() != v2.dep_tracking_hash());
    assert!(v1.dep_tracking_hash() != v3.dep_tracking_hash());
    assert!(v2.dep_tracking_hash() != v3.dep_tracking_hash());

    // Check clone
    assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
    assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
    assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
}

#[test]
fn test_output_types_tracking_hash_different_construction_order() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();

    v1.output_types = OutputTypes::new(&[
        (OutputType::Exe, Some(PathBuf::from("./some/thing"))),
        (OutputType::Bitcode, Some(PathBuf::from("./some/thing.bc"))),
    ]);

    v2.output_types = OutputTypes::new(&[
        (OutputType::Bitcode, Some(PathBuf::from("./some/thing.bc"))),
        (OutputType::Exe, Some(PathBuf::from("./some/thing"))),
    ]);

    assert_eq!(v1.dep_tracking_hash(), v2.dep_tracking_hash());

    // Check clone
    assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
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

    assert_eq!(v1.dep_tracking_hash(), v2.dep_tracking_hash());
    assert_eq!(v1.dep_tracking_hash(), v3.dep_tracking_hash());
    assert_eq!(v2.dep_tracking_hash(), v3.dep_tracking_hash());

    // Check clone
    assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
    assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
    assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
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

    assert!(v1.dep_tracking_hash() != v2.dep_tracking_hash());
    assert!(v1.dep_tracking_hash() != v3.dep_tracking_hash());
    assert!(v2.dep_tracking_hash() != v3.dep_tracking_hash());

    // Check clone
    assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
    assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
    assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
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

    assert_eq!(v1.dep_tracking_hash(), v2.dep_tracking_hash());

    // Check clone
    assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
    assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
}

#[test]
fn test_search_paths_tracking_hash_different_order() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();
    let mut v4 = Options::default();

    const JSON: ErrorOutputType = ErrorOutputType::Json {
        pretty: false,
        json_rendered: HumanReadableErrorType::Default(ColorConfig::Never),
    };

    // Reference
    v1.search_paths.push(SearchPath::from_cli_opt("native=abc", JSON));
    v1.search_paths.push(SearchPath::from_cli_opt("crate=def", JSON));
    v1.search_paths.push(SearchPath::from_cli_opt("dependency=ghi", JSON));
    v1.search_paths.push(SearchPath::from_cli_opt("framework=jkl", JSON));
    v1.search_paths.push(SearchPath::from_cli_opt("all=mno", JSON));

    v2.search_paths.push(SearchPath::from_cli_opt("native=abc", JSON));
    v2.search_paths.push(SearchPath::from_cli_opt("dependency=ghi", JSON));
    v2.search_paths.push(SearchPath::from_cli_opt("crate=def", JSON));
    v2.search_paths.push(SearchPath::from_cli_opt("framework=jkl", JSON));
    v2.search_paths.push(SearchPath::from_cli_opt("all=mno", JSON));

    v3.search_paths.push(SearchPath::from_cli_opt("crate=def", JSON));
    v3.search_paths.push(SearchPath::from_cli_opt("framework=jkl", JSON));
    v3.search_paths.push(SearchPath::from_cli_opt("native=abc", JSON));
    v3.search_paths.push(SearchPath::from_cli_opt("dependency=ghi", JSON));
    v3.search_paths.push(SearchPath::from_cli_opt("all=mno", JSON));

    v4.search_paths.push(SearchPath::from_cli_opt("all=mno", JSON));
    v4.search_paths.push(SearchPath::from_cli_opt("native=abc", JSON));
    v4.search_paths.push(SearchPath::from_cli_opt("crate=def", JSON));
    v4.search_paths.push(SearchPath::from_cli_opt("dependency=ghi", JSON));
    v4.search_paths.push(SearchPath::from_cli_opt("framework=jkl", JSON));

    assert!(v1.dep_tracking_hash() == v2.dep_tracking_hash());
    assert!(v1.dep_tracking_hash() == v3.dep_tracking_hash());
    assert!(v1.dep_tracking_hash() == v4.dep_tracking_hash());

    // Check clone
    assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
    assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
    assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
    assert_eq!(v4.dep_tracking_hash(), v4.clone().dep_tracking_hash());
}

#[test]
fn test_native_libs_tracking_hash_different_values() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();
    let mut v4 = Options::default();

    // Reference
    v1.libs = vec![
        (String::from("a"), None, Some(cstore::NativeStatic)),
        (String::from("b"), None, Some(cstore::NativeFramework)),
        (String::from("c"), None, Some(cstore::NativeUnknown)),
    ];

    // Change label
    v2.libs = vec![
        (String::from("a"), None, Some(cstore::NativeStatic)),
        (String::from("X"), None, Some(cstore::NativeFramework)),
        (String::from("c"), None, Some(cstore::NativeUnknown)),
    ];

    // Change kind
    v3.libs = vec![
        (String::from("a"), None, Some(cstore::NativeStatic)),
        (String::from("b"), None, Some(cstore::NativeStatic)),
        (String::from("c"), None, Some(cstore::NativeUnknown)),
    ];

    // Change new-name
    v4.libs = vec![
        (String::from("a"), None, Some(cstore::NativeStatic)),
        (String::from("b"), Some(String::from("X")), Some(cstore::NativeFramework)),
        (String::from("c"), None, Some(cstore::NativeUnknown)),
    ];

    assert!(v1.dep_tracking_hash() != v2.dep_tracking_hash());
    assert!(v1.dep_tracking_hash() != v3.dep_tracking_hash());
    assert!(v1.dep_tracking_hash() != v4.dep_tracking_hash());

    // Check clone
    assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
    assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
    assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
    assert_eq!(v4.dep_tracking_hash(), v4.clone().dep_tracking_hash());
}

#[test]
fn test_native_libs_tracking_hash_different_order() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();

    // Reference
    v1.libs = vec![
        (String::from("a"), None, Some(cstore::NativeStatic)),
        (String::from("b"), None, Some(cstore::NativeFramework)),
        (String::from("c"), None, Some(cstore::NativeUnknown)),
    ];

    v2.libs = vec![
        (String::from("b"), None, Some(cstore::NativeFramework)),
        (String::from("a"), None, Some(cstore::NativeStatic)),
        (String::from("c"), None, Some(cstore::NativeUnknown)),
    ];

    v3.libs = vec![
        (String::from("c"), None, Some(cstore::NativeUnknown)),
        (String::from("a"), None, Some(cstore::NativeStatic)),
        (String::from("b"), None, Some(cstore::NativeFramework)),
    ];

    assert!(v1.dep_tracking_hash() == v2.dep_tracking_hash());
    assert!(v1.dep_tracking_hash() == v3.dep_tracking_hash());
    assert!(v2.dep_tracking_hash() == v3.dep_tracking_hash());

    // Check clone
    assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
    assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
    assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
}

#[test]
fn test_codegen_options_tracking_hash() {
    let reference = Options::default();
    let mut opts = Options::default();

    // Make sure that changing an [UNTRACKED] option leaves the hash unchanged.
    // This list is in alphabetical order.

    opts.cg.ar = String::from("abc");
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.codegen_units = Some(42);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.default_linker_libraries = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.extra_filename = String::from("extra-filename");
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.incremental = Some(String::from("abc"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    // `link_arg` is omitted because it just forwards to `link_args`.

    opts.cg.link_args = vec![String::from("abc"), String::from("def")];
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.link_dead_code = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.linker = Some(PathBuf::from("linker"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.linker_flavor = Some(LinkerFlavor::Gcc);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.no_stack_check = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.remark = Passes::Some(vec![String::from("pass1"), String::from("pass2")]);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.rpath = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.save_temps = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    // Make sure that changing a [TRACKED] option changes the hash.
    // This list is in alphabetical order.

    opts = reference.clone();
    opts.cg.bitcode_in_rlib = false;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.code_model = Some(String::from("code model"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.debug_assertions = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.debuginfo = 0xdeadbeef;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.force_frame_pointers = Some(false);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.inline_threshold = Some(0xf007ba11);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.linker_plugin_lto = LinkerPluginLto::LinkerPluginAuto;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.llvm_args = vec![String::from("1"), String::from("2")];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.lto = LtoCli::Fat;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.metadata = vec![String::from("A"), String::from("B")];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_prepopulate_passes = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_redzone = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_vectorize_loops = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_vectorize_slp = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.opt_level = "3".to_string();
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.overflow_checks = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.panic = Some(PanicStrategy::Abort);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.passes = vec![String::from("1"), String::from("2")];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.prefer_dynamic = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.profile_generate = SwitchWithOptPath::Enabled(None);
    assert_ne!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.profile_use = Some(PathBuf::from("abc"));
    assert_ne!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.relocation_model = Some(String::from("relocation model"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.soft_float = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.target_cpu = Some(String::from("abc"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.target_feature = String::from("all the features, all of them");
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());
}

#[test]
fn test_debugging_options_tracking_hash() {
    let reference = Options::default();
    let mut opts = Options::default();

    // Make sure that changing an [UNTRACKED] option leaves the hash unchanged.
    // This list is in alphabetical order.

    opts.debugging_opts.ast_json = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.ast_json_noexpand = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.borrowck = String::from("other");
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.borrowck_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.control_flow_guard = CFGuard::Checks;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.deduplicate_diagnostics = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.dep_tasks = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.dont_buffer_diagnostics = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.dump_dep_graph = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.dump_mir = Some(String::from("abc"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.dump_mir_dataflow = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.dump_mir_dir = String::from("abc");
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.dump_mir_exclude_pass_number = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.dump_mir_graphviz = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.emit_stack_sizes = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.hir_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.identify_regions = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.incremental_ignore_spans = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.incremental_info = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.incremental_verify_ich = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.input_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.keep_hygiene_data = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.link_native_libraries = false;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.llvm_time_trace = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.ls = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.macro_backtrace = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.meta_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.nll_facts = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.no_analysis = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.no_interleave_lints = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.no_leak_check = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.no_parallel_llvm = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.parse_only = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.perf_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.polonius = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    // `pre_link_arg` is omitted because it just forwards to `pre_link_args`.

    opts.debugging_opts.pre_link_args = vec![String::from("abc"), String::from("def")];
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.print_link_args = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.print_llvm_passes = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.print_mono_items = Some(String::from("abc"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.print_region_graph = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.print_type_sizes = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.query_dep_graph = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.query_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.save_analysis = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.self_profile = SwitchWithOptPath::Enabled(None);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.self_profile_events = Some(vec![String::new()]);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.span_free_formats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.terminal_width = Some(80);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.threads = 99;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.time = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.time_llvm_passes = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.time_passes = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.trace_macros = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.ui_testing = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.unpretty = Some("expanded".to_string());
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.unstable_options = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.debugging_opts.verbose = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    // Make sure that changing a [TRACKED] option changes the hash.
    // This list is in alphabetical order.

    opts = reference.clone();
    opts.debugging_opts.allow_features = Some(vec![String::from("lang_items")]);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.always_encode_mir = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.asm_comments = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.binary_dep_depinfo = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.codegen_backend = Some("abc".to_string());
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.crate_attr = vec!["abc".to_string()];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.debug_macros = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.dep_info_omit_d_target = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.dual_proc_macros = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.embed_bitcode = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.fewer_names = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.force_overflow_checks = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.force_unstable_if_unmarked = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.fuel = Some(("abc".to_string(), 99));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.human_readable_cgu_names = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.inline_in_all_cgus = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.insert_sideeffect = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.instrument_mcount = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.link_only = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.merge_functions = Some(MergeFunctions::Disabled);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.mir_emit_retag = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.mir_opt_level = 3;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.mutable_noalias = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.new_llvm_pass_manager = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.no_codegen = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.no_generate_arange_section = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.no_landing_pads = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.no_link = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.no_profiler_runtime = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.osx_rpath_install_name = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.panic_abort_tests = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.plt = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.print_fuel = Some("abc".to_string());
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.profile = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.relro_level = Some(RelroLevel::Full);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.report_delayed_bugs = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.run_dsymutil = false;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.sanitizer = Some(Sanitizer::Address);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.sanitizer_memory_track_origins = 2;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.sanitizer_recover = vec![Sanitizer::Address];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.saturating_float_casts = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.share_generics = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.show_span = Some(String::from("abc"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.src_hash_algorithm = Some(SourceFileHashAlgorithm::Sha1);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.strip_debuginfo_if_disabled = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.symbol_mangling_version = SymbolManglingVersion::V0;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.teach = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.thinlto = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.tls_model = Some(String::from("tls model"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.treat_err_as_bug = Some(1);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.unleash_the_miri_inside_of_you = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.verify_llvm_ir = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());
}

#[test]
fn test_edition_parsing() {
    // test default edition
    let options = Options::default();
    assert!(options.edition == DEFAULT_EDITION);

    let matches = optgroups().parse(&["--edition=2018".to_string()]).unwrap();
    let (sessopts, _) = build_session_options_and_crate_config(matches);
    assert!(sessopts.edition == Edition::Edition2018)
}
