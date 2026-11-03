//! Extracts the list of target feature implications from rustc_target to a generated file that rust-analyzer can use for implications.

use std::{collections::BTreeSet, fs, iter::repeat};

use itertools::Itertools;
use xshell::Shell;

use crate::{
    codegen::{add_preamble, clone_rust, ensure_file_contents, reformat},
    project_root,
};

const DESTINATION: &str = "crates/hir-ty/src/target_feature/generated.rs";

/// This clones the rust repo, and so is not worth to keep up-to-date on a constant basis.
pub(crate) fn generate(check: bool) {
    let sh = &Shell::new().unwrap();
    let rust_repo = clone_rust(sh);
    let rustc_target_contents =
        fs::read_to_string(rust_repo.join("compiler/rustc_target/src/target_features.rs")).unwrap();

    let mut contents = String::from(
        r#"
// spellchecker:off

#[expect(dead_code, reason = "stubs for rustc_target")]
pub(super) enum Stability {
    Stable,
    Unstable(()),
    InternalOnly { reason: &'static str, hard_error: bool },
}
use Stability::*;

type ImpliedFeatures = &'static [&'static str];
    "#,
    );

    // Identifiers of all <ARCH>_FEATURES arrays in rustc_target
    let mut array_names = vec![];
    // Identifiers of all Unstable(sym::<unstable feature gate>) feature gates
    let mut unstable_feature_gates = BTreeSet::new();

    // Whether or not the line iterator is currently in the middle of a feature list array
    let mut currently_in_feature_list = false;
    // Go through the file and do some manual parsing to keep only the lines that are part of a target feature array.
    let target_feature_arrays = rustc_target_contents
        .lines()
        .filter(|line| {
            if line.contains("&[(&str, Stability, ImpliedFeatures)]") {
                // Start of a new feature list
                assert!(!currently_in_feature_list);
                currently_in_feature_list = true;
                array_names.push(extract_list_ident_to_array_entry(line).unwrap());
            } else if currently_in_feature_list && *line == "];" {
                // This is the end of a feature list, so ignore all lines after this one until the start of the next feature list
                currently_in_feature_list = false;
                return true;
            }

            if currently_in_feature_list {
                // Keep this line and maybe create a stub feature gate
                unstable_feature_gates.extend(extract_to_unstable_feature_gate(line));
                true
            } else {
                false
            }
        })
        .interleave_shortest(repeat("\n"));
    contents.extend(target_feature_arrays);

    let mut implications_raw = String::from(
        "pub(super) const TARGET_FEATURE_IMPLICATIONS_RAW: &[&[(&str, Stability, ImpliedFeatures)]] = &[\n",
    );
    implications_raw.extend(array_names);
    implications_raw.push_str("];\n");
    contents.push_str(&implications_raw);

    let mut sym = String::from(
        r#"
#[expect(non_upper_case_globals, reason = "stubs for rustc_target")]
mod sym {
"#,
    );
    sym.extend(unstable_feature_gates);
    sym.push_str("}\n// spellchecker:on\n");
    contents.push_str(&sym);

    let contents = add_preamble(crate::flags::CodegenType::TargetFeatures, reformat(contents));

    let destination = project_root().join(DESTINATION);
    ensure_file_contents(
        crate::flags::CodegenType::TargetFeatures,
        destination.as_path(),
        &contents,
        check,
    );
}

/// Extract the identifier of a feature list `const $ARCH_FEATURES:` or `static $ARCH_FEATURES:` to a list entry `$ARCH_FEATURES,`
fn extract_list_ident_to_array_entry(line: &str) -> Option<String> {
    let ident = line.split_once(' ')?.1.split_once(':')?.0;
    Some(format!("{ident},\n"))
}

/// Extract the identifier from `Unstable(sym::$IDENT)` and create a new mock identifier with it
fn extract_to_unstable_feature_gate(line: &str) -> Option<String> {
    let ident = line.split_once("Unstable(sym::")?.1.split_once(')')?.0;

    Some(format!("pub(in super::super) const {ident}: () = ();\n"))
}
