//! Tidy check to ensure paths mentioned in triagebot.toml exist in the project.

use std::path::Path;

use toml::Value;

use crate::diagnostics::TidyCtx;

pub fn check(path: &Path, tidy_ctx: TidyCtx) {
    let mut check = tidy_ctx.start_check("triagebot");
    let triagebot_path = path.join("triagebot.toml");

    // This check is mostly to catch broken path filters *within* `triagebot.toml`, and not enforce
    // the existence of `triagebot.toml` itself (which is more obvious), as distribution tarballs
    // will not include non-essential bits like `triagebot.toml`.
    if !triagebot_path.exists() {
        return;
    }

    let contents = std::fs::read_to_string(&triagebot_path).unwrap();
    let config: Value = toml::from_str(&contents).unwrap();

    // Check [mentions."*"] sections, i.e. [mentions."compiler/rustc_const_eval/src/"]
    if let Some(Value::Table(mentions)) = config.get("mentions") {
        for (entry_key, entry_val) in mentions.iter() {
            // If the type is set to something other than "filename", then this is not a path.
            if entry_val.get("type").is_some_and(|t| t.as_str().unwrap_or_default() != "filename") {
                continue;
            }
            let path_str = entry_key;
            // Remove quotes from the path
            let clean_path = path_str.trim_matches('"');
            let full_path = path.join(clean_path);

            if !full_path.exists() {
                check.error(format!(
                    "triagebot.toml [mentions.*] contains path '{clean_path}' which doesn't exist"
                ));
            }
        }
    } else {
        check.error(
            "triagebot.toml missing [mentions.*] section, this wrong for rust-lang/rust repo.",
        );
    }

    // Check [assign.owners] sections, i.e.
    // [assign.owners]
    // "/.github/workflows" = ["infra-ci"]
    if let Some(Value::Table(assign)) = config.get("assign") {
        if let Some(Value::Table(owners)) = assign.get("owners") {
            for path_str in owners.keys() {
                // Remove quotes and leading slash from the path
                let clean_path = path_str.trim_matches('"').trim_start_matches('/');
                let full_path = path.join(clean_path);

                if !full_path.exists() {
                    check.error(format!(
                        "triagebot.toml [assign.owners] contains path '{clean_path}' which doesn't exist"
                    ));
                }
            }
        } else {
            check.error(
                "triagebot.toml missing [assign.owners] section, this wrong for rust-lang/rust repo."
            );
        }
    }

    // Verify that trigger_files in [autolabel."*"] exist in the project, i.e.
    // [autolabel."A-rustdoc-search"]
    // trigger_files = [
    //    "src/librustdoc/html/static/js/search.js",
    //    "tests/rustdoc-js",
    //    "tests/rustdoc-js-std",
    // ]
    if let Some(Value::Table(autolabels)) = config.get("autolabel") {
        for (label, content) in autolabels {
            if let Some(trigger_files) = content.get("trigger_files").and_then(|v| v.as_array()) {
                for file in trigger_files {
                    if let Some(file_str) = file.as_str() {
                        let full_path = path.join(file_str);

                        // Handle both file and directory paths
                        if !full_path.exists() {
                            check.error(format!(
                                "triagebot.toml [autolabel.{label}] contains trigger_files path '{file_str}' which doesn't exist",
                            ));
                        }
                    }
                }
            }
        }
    }
}
