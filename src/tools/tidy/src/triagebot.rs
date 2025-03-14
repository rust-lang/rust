//! Tidy check to ensure paths mentioned in triagebot.toml exist in the project.

use std::path::Path;

use toml::Value;

pub fn check(path: &Path, bad: &mut bool) {
    let triagebot_path = path.join("triagebot.toml");
    if !triagebot_path.exists() {
        tidy_error!(bad, "triagebot.toml file not found");
        return;
    }

    let contents = std::fs::read_to_string(&triagebot_path).unwrap();
    let config: Value = toml::from_str(&contents).unwrap();

    // Check [mentions."*"] sections, i.e. [mentions."compiler/rustc_const_eval/src/"]
    if let Some(Value::Table(mentions)) = config.get("mentions") {
        for path_str in mentions.keys() {
            // Remove quotes from the path
            let clean_path = path_str.trim_matches('"');
            let full_path = path.join(clean_path);

            if !full_path.exists() {
                tidy_error!(
                    bad,
                    "triagebot.toml [mentions.*] contains path '{}' which doesn't exist",
                    clean_path
                );
            }
        }
    } else {
        tidy_error!(
            bad,
            "triagebot.toml missing [mentions.*] section, this wrong for rust-lang/rust repo."
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
                    tidy_error!(
                        bad,
                        "triagebot.toml [assign.owners] contains path '{}' which doesn't exist",
                        clean_path
                    );
                }
            }
        } else {
            tidy_error!(
                bad,
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
                            tidy_error!(
                                bad,
                                "triagebot.toml [autolabel.{}] contains trigger_files path '{}' which doesn't exist",
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
