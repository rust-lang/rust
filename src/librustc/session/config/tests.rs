use getopts;
use crate::lint;
use crate::middle::cstore;
use crate::session::config::{
    build_configuration,
    build_session_options_and_crate_config,
    to_crate_config
};
use crate::session::config::{LtoCli, LinkerPluginLto, SwitchWithOptPath, ExternEntry};
use crate::session::build_session;
use crate::session::search_paths::SearchPath;
use std::collections::{BTreeMap, BTreeSet};
use std::iter::FromIterator;
use std::path::PathBuf;
use super::{Externs, OutputType, OutputTypes, SymbolManglingVersion};
use rustc_target::spec::{MergeFunctions, PanicStrategy, RelroLevel};
use syntax::symbol::sym;
use syntax::edition::{Edition, DEFAULT_EDITION};
use syntax;
use super::Options;

impl ExternEntry {
    fn new_public<S: Into<String>,
                  I: IntoIterator<Item = Option<S>>>(locations: I) -> ExternEntry {
        let locations: BTreeSet<_> = locations.into_iter().map(|o| o.map(|s| s.into()))
            .collect();

        ExternEntry {
            locations,
            is_private_dep: false
        }
    }
}

fn optgroups() -> getopts::Options {
    let mut opts = getopts::Options::new();
    for group in super::rustc_optgroups() {
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
    syntax::with_default_globals(|| {
        let matches = &match optgroups().parse(&["--test".to_string()]) {
            Ok(m) => m,
            Err(f) => panic!("test_switch_implies_cfg_test: {}", f),
        };
        let registry = errors::registry::Registry::new(&[]);
        let (sessopts, cfg) = build_session_options_and_crate_config(matches);
        let sess = build_session(sessopts, None, registry);
        let cfg = build_configuration(&sess, to_crate_config(cfg));
        assert!(cfg.contains(&(sym::test, None)));
    });
}

// When the user supplies --test and --cfg test, don't implicitly add
// another --cfg test
#[test]
fn test_switch_implies_cfg_test_unless_cfg_test() {
    syntax::with_default_globals(|| {
        let matches = &match optgroups().parse(&["--test".to_string(),
                                                 "--cfg=test".to_string()]) {
            Ok(m) => m,
            Err(f) => panic!("test_switch_implies_cfg_test_unless_cfg_test: {}", f),
        };
        let registry = errors::registry::Registry::new(&[]);
        let (sessopts, cfg) = build_session_options_and_crate_config(matches);
        let sess = build_session(sessopts, None, registry);
        let cfg = build_configuration(&sess, to_crate_config(cfg));
        let mut test_items = cfg.iter().filter(|&&(name, _)| name == sym::test);
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    });
}

#[test]
fn test_can_print_warnings() {
    syntax::with_default_globals(|| {
        let matches = optgroups().parse(&["-Awarnings".to_string()]).unwrap();
        let registry = errors::registry::Registry::new(&[]);
        let (sessopts, _) = build_session_options_and_crate_config(&matches);
        let sess = build_session(sessopts, None, registry);
        assert!(!sess.diagnostic().flags.can_emit_warnings);
    });

    syntax::with_default_globals(|| {
        let matches = optgroups()
            .parse(&["-Awarnings".to_string(), "-Dwarnings".to_string()])
            .unwrap();
        let registry = errors::registry::Registry::new(&[]);
        let (sessopts, _) = build_session_options_and_crate_config(&matches);
        let sess = build_session(sessopts, None, registry);
        assert!(sess.diagnostic().flags.can_emit_warnings);
    });

    syntax::with_default_globals(|| {
        let matches = optgroups().parse(&["-Adead_code".to_string()]).unwrap();
        let registry = errors::registry::Registry::new(&[]);
        let (sessopts, _) = build_session_options_and_crate_config(&matches);
        let sess = build_session(sessopts, None, registry);
        assert!(sess.diagnostic().flags.can_emit_warnings);
    });
}

#[test]
fn test_output_types_tracking_hash_different_paths() {
    let mut v1 = Options::default();
    let mut v2 = Options::default();
    let mut v3 = Options::default();

    v1.output_types =
        OutputTypes::new(&[(OutputType::Exe, Some(PathBuf::from("./some/thing")))]);
    v2.output_types =
        OutputTypes::new(&[(OutputType::Exe, Some(PathBuf::from("/some/thing")))]);
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
        (
            String::from("a"),
            ExternEntry::new_public(vec![Some("b"), Some("c")])
        ),
        (
            String::from("d"),
            ExternEntry::new_public(vec![Some("e"), Some("f")])
        ),
    ]));

    v2.externs = Externs::new(mk_map(vec![
        (
            String::from("d"),
            ExternEntry::new_public(vec![Some("e"), Some("f")])
        ),
        (
            String::from("a"),
            ExternEntry::new_public(vec![Some("b"), Some("c")])
        ),
    ]));

    v3.externs = Externs::new(mk_map(vec![
        (
            String::from("a"),
            ExternEntry::new_public(vec![Some("b"), Some("c")])
        ),
        (
            String::from("d"),
            ExternEntry::new_public(vec![Some("f"), Some("e")])
        ),
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
        (String::from("a"), lint::Allow),
        (String::from("b"), lint::Warn),
        (String::from("c"), lint::Deny),
        (String::from("d"), lint::Forbid),
    ];

    v2.lint_opts = vec![
        (String::from("a"), lint::Allow),
        (String::from("b"), lint::Warn),
        (String::from("X"), lint::Deny),
        (String::from("d"), lint::Forbid),
    ];

    v3.lint_opts = vec![
        (String::from("a"), lint::Allow),
        (String::from("b"), lint::Warn),
        (String::from("c"), lint::Forbid),
        (String::from("d"), lint::Deny),
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
        (String::from("a"), lint::Allow),
        (String::from("b"), lint::Warn),
        (String::from("c"), lint::Deny),
        (String::from("d"), lint::Forbid),
    ];

    v2.lint_opts = vec![
        (String::from("a"), lint::Allow),
        (String::from("c"), lint::Deny),
        (String::from("b"), lint::Warn),
        (String::from("d"), lint::Forbid),
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

    const JSON: super::ErrorOutputType = super::ErrorOutputType::Json {
        pretty: false,
        json_rendered: super::HumanReadableErrorType::Default(super::ColorConfig::Never),
    };

    // Reference
    v1.search_paths
        .push(SearchPath::from_cli_opt("native=abc", JSON));
    v1.search_paths
        .push(SearchPath::from_cli_opt("crate=def", JSON));
    v1.search_paths
        .push(SearchPath::from_cli_opt("dependency=ghi", JSON));
    v1.search_paths
        .push(SearchPath::from_cli_opt("framework=jkl", JSON));
    v1.search_paths
        .push(SearchPath::from_cli_opt("all=mno", JSON));

    v2.search_paths
        .push(SearchPath::from_cli_opt("native=abc", JSON));
    v2.search_paths
        .push(SearchPath::from_cli_opt("dependency=ghi", JSON));
    v2.search_paths
        .push(SearchPath::from_cli_opt("crate=def", JSON));
    v2.search_paths
        .push(SearchPath::from_cli_opt("framework=jkl", JSON));
    v2.search_paths
        .push(SearchPath::from_cli_opt("all=mno", JSON));

    v3.search_paths
        .push(SearchPath::from_cli_opt("crate=def", JSON));
    v3.search_paths
        .push(SearchPath::from_cli_opt("framework=jkl", JSON));
    v3.search_paths
        .push(SearchPath::from_cli_opt("native=abc", JSON));
    v3.search_paths
        .push(SearchPath::from_cli_opt("dependency=ghi", JSON));
    v3.search_paths
        .push(SearchPath::from_cli_opt("all=mno", JSON));

    v4.search_paths
        .push(SearchPath::from_cli_opt("all=mno", JSON));
    v4.search_paths
        .push(SearchPath::from_cli_opt("native=abc", JSON));
    v4.search_paths
        .push(SearchPath::from_cli_opt("crate=def", JSON));
    v4.search_paths
        .push(SearchPath::from_cli_opt("dependency=ghi", JSON));
    v4.search_paths
        .push(SearchPath::from_cli_opt("framework=jkl", JSON));

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
        (
            String::from("b"),
            Some(String::from("X")),
            Some(cstore::NativeFramework),
        ),
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

    // Make sure the changing an [UNTRACKED] option leaves the hash unchanged
    opts.cg.ar = Some(String::from("abc"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.linker = Some(PathBuf::from("linker"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.link_args = Some(vec![String::from("abc"), String::from("def")]);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.link_dead_code = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.rpath = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.extra_filename = String::from("extra-filename");
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.codegen_units = Some(42);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.remark = super::Passes::Some(vec![String::from("pass1"), String::from("pass2")]);
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.save_temps = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts.cg.incremental = Some(String::from("abc"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    // Make sure changing a [TRACKED] option changes the hash
    opts = reference.clone();
    opts.cg.lto = LtoCli::Fat;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.target_cpu = Some(String::from("abc"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.target_feature = String::from("all the features, all of them");
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.passes = vec![String::from("1"), String::from("2")];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.llvm_args = vec![String::from("1"), String::from("2")];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.overflow_checks = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_prepopulate_passes = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_vectorize_loops = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_vectorize_slp = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.soft_float = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.prefer_dynamic = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_integrated_as = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.no_redzone = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.relocation_model = Some(String::from("relocation model"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.code_model = Some(String::from("code model"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.tls_model = Some(String::from("tls model"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.profile_generate = SwitchWithOptPath::Enabled(None);
    assert_ne!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.profile_use = Some(PathBuf::from("abc"));
    assert_ne!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.metadata = vec![String::from("A"), String::from("B")];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.debuginfo = Some(0xdeadbeef);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.debuginfo = Some(0xba5eba11);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.force_frame_pointers = Some(false);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.debug_assertions = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.inline_threshold = Some(0xf007ba11);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.panic = Some(PanicStrategy::Abort);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.cg.linker_plugin_lto = LinkerPluginLto::LinkerPluginAuto;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());
}

#[test]
fn test_debugging_options_tracking_hash() {
    let reference = Options::default();
    let mut opts = Options::default();

    // Make sure the changing an [UNTRACKED] option leaves the hash unchanged
    opts.debugging_opts.verbose = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.time_passes = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.time_llvm_passes = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.input_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.borrowck_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.meta_stats = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.print_link_args = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.print_llvm_passes = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.ast_json = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.ast_json_noexpand = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.ls = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.save_analysis = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.flowgraph_print_loans = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.flowgraph_print_moves = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.flowgraph_print_assigns = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.flowgraph_print_all = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.print_region_graph = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.parse_only = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.incremental = Some(String::from("abc"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.dump_dep_graph = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.query_dep_graph = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.no_analysis = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.unstable_options = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.trace_macros = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.keep_hygiene_data = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.keep_ast = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.print_mono_items = Some(String::from("abc"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.dump_mir = Some(String::from("abc"));
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.dump_mir_dir = String::from("abc");
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
    opts.debugging_opts.dump_mir_graphviz = true;
    assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

    // Make sure changing a [TRACKED] option changes the hash
    opts = reference.clone();
    opts.debugging_opts.asm_comments = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.verify_llvm_ir = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.no_landing_pads = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.fewer_names = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.no_codegen = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.treat_err_as_bug = Some(1);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.report_delayed_bugs = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.continue_parse_after_error = true;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.extra_plugins = vec![String::from("plugin1"), String::from("plugin2")];
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.force_overflow_checks = Some(true);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.show_span = Some(String::from("abc"));
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.mir_opt_level = 3;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.relro_level = Some(RelroLevel::Full);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.merge_functions = Some(MergeFunctions::Disabled);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.allow_features = Some(vec![String::from("lang_items")]);
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

    opts = reference.clone();
    opts.debugging_opts.symbol_mangling_version = SymbolManglingVersion::V0;
    assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());
}

#[test]
fn test_edition_parsing() {
    // test default edition
    let options = Options::default();
    assert!(options.edition == DEFAULT_EDITION);

    let matches = optgroups()
        .parse(&["--edition=2018".to_string()])
        .unwrap();
    let (sessopts, _) = build_session_options_and_crate_config(&matches);
    assert!(sessopts.edition == Edition::Edition2018)
}
