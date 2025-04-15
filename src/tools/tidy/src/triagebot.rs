//! Tidy check to ensure paths mentioned in triagebot.toml exist in the project.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use build_helper::ci::CiEnv;
use toml::Value;

pub fn check(checkout_root: &Path, bad: &mut bool) {
    let triagebot_path = checkout_root.join("triagebot.toml");
    if !triagebot_path.exists() {
        tidy_error!(bad, "triagebot.toml file not found");
        return;
    }

    let contents = std::fs::read_to_string(&triagebot_path).unwrap();
    let config: Value = toml::from_str(&contents).unwrap();

    // Cache mapping between submodule path <-> whether submodule is checked out. This cache is to
    // avoid excessive filesystem accesses.
    let submodule_checked_out_status = cache_submodule_checkout_status(checkout_root);

    // Check `[mentions."*"]` sections, i.e.
    //
    // ```
    // [mentions."compiler/rustc_const_eval/src/"]
    // ```
    if let Some(Value::Table(mentions)) = config.get("mentions") {
        for path_str in mentions.keys() {
            // Remove quotes from the path
            let clean_path = path_str.trim_matches('"');
            let full_path = checkout_root.join(clean_path);

            if !check_path_exists_if_required(&submodule_checked_out_status, &full_path) {
                tidy_error!(
                    bad,
                    "triagebot.toml `[mentions.*]` contains path `{}` which doesn't exist",
                    clean_path
                );
            }
        }
    } else {
        tidy_error!(
            bad,
            "`triagebot.toml` is missing the `[mentions.*]` section; this is wrong for the \
            `rust-lang/rust` repo."
        );
    }

    // Check `[assign.owners]` sections, i.e.
    //
    // ```
    // [assign.owners]
    // "/.github/workflows" = ["infra-ci"]
    // ```
    if let Some(Value::Table(assign)) = config.get("assign") {
        if let Some(Value::Table(owners)) = assign.get("owners") {
            for path_str in owners.keys() {
                // Remove quotes and leading slash from the path
                let clean_path = path_str.trim_matches('"').trim_start_matches('/');
                let full_path = checkout_root.join(clean_path);

                if !check_path_exists_if_required(&submodule_checked_out_status, &full_path) {
                    tidy_error!(
                        bad,
                        "`triagebot.toml` `[assign.owners]` contains path `{}` which doesn't exist",
                        clean_path
                    );
                }
            }
        } else {
            tidy_error!(
                bad,
                "`triagebot.toml` is missing the `[assign.owners]` section; this is wrong for the \
                `rust-lang/rust` repo."
            );
        }
    }

    // Verify that `trigger_files` paths in `[autolabel."*"]` exists, i.e.
    //
    // ```
    // [autolabel."A-rustdoc-search"]
    // trigger_files = [
    //    "src/librustdoc/html/static/js/search.js",
    //    "tests/rustdoc-js",
    //    "tests/rustdoc-js-std",
    // ]
    // ```
    if let Some(Value::Table(autolabels)) = config.get("autolabel") {
        for (label, content) in autolabels {
            if let Some(trigger_files) = content.get("trigger_files").and_then(|v| v.as_array()) {
                for file in trigger_files {
                    if let Some(file_str) = file.as_str() {
                        let full_path = checkout_root.join(file_str);

                        // Handle both file and directory paths
                        if !check_path_exists_if_required(&submodule_checked_out_status, &full_path)
                        {
                            tidy_error!(
                                bad,
                                "`triagebot.toml` `[autolabel.{}]` contains `trigger_files` path \
                                `{}` which doesn't exist",
                                label,
                                file_str
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Very naive heuristics for whether a submodule is checked out.
fn cache_submodule_checkout_status(checkout_root: &Path) -> HashMap<PathBuf, bool> {
    let mut cache = HashMap::default();

    // NOTE: can't assume `git` exists.
    let submodule_paths = build_helper::util::parse_gitmodules(&checkout_root);

    for submodule in submodule_paths {
        let full_submodule_path = checkout_root.join(submodule);

        let is_checked_out = if CiEnv::is_ci() {
            // In CI, require all submodules to be checked out and thus don't skip checking any
            // paths.
            true
        } else {
            // NOTE: for our purposes, just skip checking paths to and under a submodule if we can't
            // read its dir locally (even if this can miss broken paths).
            std::fs::read_dir(&full_submodule_path).is_ok_and(|entry| {
                // NOTE: de-initializing a submodule can leave an empty folder behind
                entry.count() > 0
            })
        };

        if let Some(_) = cache.insert(full_submodule_path.clone(), is_checked_out) {
            panic!(
                "unexpected duplicate submodule paths in `deps::WORKSPACES`: {} already in \
                submodule checkout cache",
                full_submodule_path.display()
            );
        }
    }

    cache
}

/// Check that a path exists. This is:
///
/// - Unconditionally checked under CI environment.
/// - Only checked under local environment if submodule is checked out (if candidate path points
///   under or to a submodule).
fn check_path_exists_if_required(
    submodule_checkout_status: &HashMap<PathBuf, bool>,
    candidate: &Path,
) -> bool {
    for (submodule_path, is_checked_out) in submodule_checkout_status {
        if candidate.starts_with(submodule_path) {
            if *is_checked_out {
                return candidate.exists();
            } else {
                // Not actually checked, but just skipped.
                return true;
            }
        }
    }

    candidate.exists()
}
