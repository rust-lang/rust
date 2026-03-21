use std::path::{Path, PathBuf};

use super::parse_compiler_flags;

#[test]
fn parses_unstable_options_entries() {
    let options_rs = r#"options! {
    UnstableOptions, UnstableOptionsTargetModifiers, Z_OPTIONS, dbopts, "Z", "unstable",

    #[rustc_lint_opt_deny_field_access("test attr")]
    allow_features: Option<Vec<String>> = (None, parse_opt_comma_list, [TRACKED],
        "only allow the listed language features to be enabled in code (comma separated)"),
    dump_mir: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "dump MIR state to file.\n\
        `val` is used to select which passes and functions to dump."),
    join_lines: bool = (false, parse_bool, [TRACKED],
        "join \
         continued lines"),
    help: bool = (false, parse_no_value, [UNTRACKED], "Print unstable compiler options"),
}"#;

    let features = parse_compiler_flags(options_rs, Path::new("options/unstable.rs"));

    assert!(features.contains_key("allow_features"));
    assert!(features.contains_key("dump_mir"));
    assert!(features.contains_key("join_lines"));
    assert!(!features.contains_key("help"));

    assert!(
        features["dump_mir"]
            .description
            .as_deref()
            .expect("dump_mir description should exist")
            .starts_with("dump MIR state to file.\n"),
    );
    assert_eq!(features["join_lines"].description.as_deref(), Some("join continued lines"));
    assert_eq!(
        features["allow_features"].description.as_deref(),
        Some("only allow the listed language features to be enabled in code (comma separated)"),
    );
    assert_eq!(features["allow_features"].file, PathBuf::from("options/unstable.rs"));
    assert_eq!(features["allow_features"].line, 5);
}

#[test]
fn parser_accepts_optional_trailing_metadata() {
    let options_rs = r##"options! {
    UnstableOptions, UnstableOptionsTargetModifiers, Z_OPTIONS, dbopts, "Z", "unstable",

    deprecated_flag: bool = (false, parse_no_value, [UNTRACKED], "deprecated flag",
        is_deprecated_and_do_nothing: true),
    raw_description: bool = (false, parse_no_value, [UNTRACKED], r#"raw "quoted" text"#),
}"##;

    let features = parse_compiler_flags(options_rs, Path::new("options/unstable.rs"));
    assert_eq!(features["deprecated_flag"].description.as_deref(), Some("deprecated flag"));
    assert_eq!(features["raw_description"].description.as_deref(), Some("raw \"quoted\" text"),);
}

#[test]
fn parses_real_unstable_options_file() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let options_path = manifest_dir.join("../../../compiler/rustc_session/src/options/unstable.rs");
    let options_rs = std::fs::read_to_string(&options_path).unwrap();
    let features = parse_compiler_flags(&options_rs, &options_path);

    assert!(features.contains_key("allow_features"));
    assert!(features.contains_key("dump_mir"));
    assert!(features.contains_key("unstable_options"));
    assert!(!features.contains_key("help"));
    assert!(features["dump_mir"].line > 0);
    assert!(features["dump_mir"].description.as_deref().unwrap().starts_with("dump MIR state"));
}
