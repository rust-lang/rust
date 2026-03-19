use std::path::{Path, PathBuf};

use super::{parse_compiler_flags, parse_one_entry, skip_ws_comments_and_attrs};

#[test]
fn parses_unstable_options_entries() {
    let tidy_start = "// tidy-alphabetical-start";
    let tidy_end = "// tidy-alphabetical-end";
    let options_rs = format!(
        "\n\
        options! {{\n\
        \x20   UnstableOptions, UnstableOptionsTargetModifiers, Z_OPTIONS, dbopts, \"Z\", \"unstable\",\n\
        \n\
        \x20   {tidy_start}\n\
        \x20   #[rustc_lint_opt_deny_field_access(\"test attr\")]\n\
        \x20   allow_features: Option<Vec<String>> = (None, parse_opt_comma_list, [TRACKED],\n\
        \x20       \"only allow the listed language features to be enabled in code (comma separated)\"),\n\
        \x20   dump_mir: Option<String> = (None, parse_opt_string, [UNTRACKED],\n\
        \x20       \"dump MIR state to file.\n\
        \x20       `val` is used to select which passes and functions to dump.\"),\n\
        \x20   join_lines: bool = (false, parse_bool, [TRACKED],\n\
        \x20       \"join \\\n\
        \x20        continued lines\"),\n\
        \x20   help: bool = (false, parse_no_value, [UNTRACKED], \"Print unstable compiler options\"),\n\
        \x20   {tidy_end}\n\
        }}\n"
    );

    let features = parse_compiler_flags(&options_rs, Path::new("options.rs"));

    assert!(features.contains_key("allow_features"));
    assert!(features.contains_key("dump_mir"));
    assert!(features.contains_key("join_lines"));
    assert!(!features.contains_key("help"));

    assert_eq!(
        features["dump_mir"].description.as_deref(),
        Some(
            "dump MIR state to file.\n        `val` is used to select which passes and functions to dump."
        ),
    );
    assert_eq!(features["join_lines"].description.as_deref(), Some("join continued lines"),);
    assert_eq!(features["allow_features"].file, PathBuf::from("options.rs"));
    assert_eq!(features["allow_features"].line, 7);
}

#[test]
fn parse_one_entry_skips_help_description_and_advances() {
    let section = "\
help: bool = (false, parse_no_value, [UNTRACKED], \"Print unstable compiler options\"),\n\
join_lines: bool = (false, parse_bool, [TRACKED], \"join \\\n\
    continued lines\"),\n";

    let help_entry = parse_one_entry(section, 0);
    assert_eq!(help_entry.name, "help");
    assert!(help_entry.description.is_none());

    let mut next_idx = help_entry.next_idx;
    skip_ws_comments_and_attrs(section, &mut next_idx);
    let next_entry = parse_one_entry(section, next_idx);

    assert_eq!(next_entry.name, "join_lines");
    assert_eq!(next_entry.description.as_deref(), Some("join continued lines"),);
}

#[test]
fn parse_one_entry_accepts_optional_trailing_metadata() {
    let entry = "\
deprecated_flag: bool = (false, parse_no_value, [UNTRACKED], \"deprecated flag\",\n\
    is_deprecated_and_do_nothing: true),\n";

    let parsed = parse_one_entry(entry, 0);
    assert_eq!(parsed.name, "deprecated_flag");
    assert_eq!(parsed.description.as_deref(), Some("deprecated flag"));
}

#[test]
fn parses_real_unstable_options_file() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let options_path = manifest_dir.join("../../../compiler/rustc_session/src/options.rs");
    let options_rs = std::fs::read_to_string(&options_path).unwrap();
    let features = parse_compiler_flags(&options_rs, &options_path);

    assert!(features.contains_key("allow_features"));
    assert!(features.contains_key("dump_mir"));
    assert!(features.contains_key("unstable_options"));
    assert!(!features.contains_key("help"));
    assert!(features["dump_mir"].line > 0);
    assert!(features["dump_mir"].description.as_deref().unwrap().starts_with("dump MIR state"));
}
