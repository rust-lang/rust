//! Tidy check to ensure paths mentioned in triagebot.toml exist in the project.

use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

use toml::Value;

use crate::diagnostics::TidyCtx;

static SUBMODULES: LazyLock<Vec<&'static Path>> = LazyLock::new(|| {
    // WORKSPACES doesn't list all submodules but it's contains the main at least
    crate::deps::WORKSPACES
        .iter()
        .map(|ws| ws.submodules.iter())
        .flatten()
        .map(|p| Path::new(p))
        .collect()
});

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
        let mut builder = globset::GlobSetBuilder::new();
        let mut glob_entries = Vec::new();

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
                // The full-path doesn't exists, maybe it's a glob, let's add it to the glob set builder
                // to be checked against all the file and directories in the repository.
                let trimmed_path = clean_path.trim_end_matches('/');
                builder.add(globset::Glob::new(&format!("{trimmed_path}{{,/*}}")).unwrap());
                glob_entries.push(clean_path.to_string());
            } else if is_in_submodule(Path::new(clean_path)) {
                check.error(format!(
                    "triagebot.toml [mentions.*] '{clean_path}' cannot match inside a submodule"
                ));
            }
        }

        let gs = builder.build().unwrap();

        let mut found = HashSet::new();
        let mut matches = Vec::new();

        let cloned_path = path.to_path_buf();

        // Walk the entire repository and match any entry against the remaining paths
        for entry in ignore::WalkBuilder::new(&path)
            .filter_entry(move |entry| {
                // Ignore entries inside submodules as triagebot cannot detect them
                let entry_path = entry.path().strip_prefix(&cloned_path).unwrap();
                is_not_in_submodule(entry_path)
            })
            .build()
            .flatten()
        {
            // Strip the prefix as mentions entries are always relative to the repo
            let entry_path = entry.path().strip_prefix(path).unwrap();

            // Find the matches and add them to the found set
            gs.matches_into(entry_path, &mut matches);
            found.extend(matches.iter().copied());

            // Early-exist if all the globs have been matched
            if found.len() == glob_entries.len() {
                break;
            }
        }

        for (i, clean_path) in glob_entries.iter().enumerate() {
            if !found.contains(&i) {
                check.error(format!(
                    "triagebot.toml [mentions.*] contains '{clean_path}' which doesn't match any file or directory in the repository"
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

fn is_not_in_submodule(path: &Path) -> bool {
    SUBMODULES.contains(&path) || !SUBMODULES.iter().any(|p| path.starts_with(*p))
}

fn is_in_submodule(path: &Path) -> bool {
    !SUBMODULES.contains(&path) && SUBMODULES.iter().any(|p| path.starts_with(*p))
}
