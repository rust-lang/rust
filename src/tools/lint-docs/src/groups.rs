use crate::Lint;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

static GROUP_DESCRIPTIONS: &[(&str, &str)] = &[
    ("unused", "Lints that detect things being declared but not used, or excess syntax"),
    ("rustdoc", "Rustdoc-specific lints"),
    ("rust-2018-idioms", "Lints to nudge you toward idiomatic features of Rust 2018"),
    ("nonstandard-style", "Violation of standard naming conventions"),
    ("future-incompatible", "Lints that detect code that has future-compatibility problems"),
    ("rust-2018-compatibility", "Lints used to transition code from the 2015 edition to 2018"),
];

/// Updates the documentation of lint groups.
pub(crate) fn generate_group_docs(
    lints: &[Lint],
    rustc_path: &Path,
    out_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let groups = collect_groups(rustc_path)?;
    let groups_path = out_path.join("groups.md");
    let contents = fs::read_to_string(&groups_path)
        .map_err(|e| format!("could not read {}: {}", groups_path.display(), e))?;
    let new_contents = contents.replace("{{groups-table}}", &make_groups_table(lints, &groups)?);
    // Delete the output because rustbuild uses hard links in its copies.
    let _ = fs::remove_file(&groups_path);
    fs::write(&groups_path, new_contents)
        .map_err(|e| format!("could not write to {}: {}", groups_path.display(), e))?;
    Ok(())
}

type LintGroups = BTreeMap<String, BTreeSet<String>>;

/// Collects the group names from rustc.
fn collect_groups(rustc: &Path) -> Result<LintGroups, Box<dyn Error>> {
    let mut result = BTreeMap::new();
    let mut cmd = Command::new(rustc);
    cmd.arg("-Whelp");
    let output = cmd.output().map_err(|e| format!("failed to run command {:?}\n{}", cmd, e))?;
    if !output.status.success() {
        return Err(format!(
            "failed to collect lint info: {:?}\n--- stderr\n{}--- stdout\n{}\n",
            output.status,
            std::str::from_utf8(&output.stderr).unwrap(),
            std::str::from_utf8(&output.stdout).unwrap(),
        )
        .into());
    }
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    let lines = stdout.lines();
    let group_start = lines.skip_while(|line| !line.contains("groups provided")).skip(1);
    let table_start = group_start.skip_while(|line| !line.contains("----")).skip(1);
    for line in table_start {
        if line.is_empty() {
            break;
        }
        let mut parts = line.trim().splitn(2, ' ');
        let name = parts.next().expect("name in group");
        if name == "warnings" {
            // This is special.
            continue;
        }
        let lints =
            parts.next().ok_or_else(|| format!("expected lints following name, got `{}`", line))?;
        let lints = lints.split(',').map(|l| l.trim().to_string()).collect();
        assert!(result.insert(name.to_string(), lints).is_none());
    }
    if result.is_empty() {
        return Err(
            format!("expected at least one group in -Whelp output, got:\n{}", stdout).into()
        );
    }
    Ok(result)
}

fn make_groups_table(lints: &[Lint], groups: &LintGroups) -> Result<String, Box<dyn Error>> {
    let mut result = String::new();
    let mut to_link = Vec::new();
    result.push_str("| Group | Description | Lints |\n");
    result.push_str("|-------|-------------|-------|\n");
    result.push_str("| warnings | All lints that are set to issue warnings | See [warn-by-default] for the default set of warnings |\n");
    for (group_name, group_lints) in groups {
        let description = GROUP_DESCRIPTIONS.iter().find(|(n, _)| n == group_name)
            .ok_or_else(|| format!("lint group `{}` does not have a description, please update the GROUP_DESCRIPTIONS list", group_name))?
            .1;
        to_link.extend(group_lints);
        let brackets: Vec<_> = group_lints.iter().map(|l| format!("[{}]", l)).collect();
        write!(result, "| {} | {} | {} |\n", group_name, description, brackets.join(", ")).unwrap();
    }
    result.push('\n');
    result.push_str("[warn-by-default]: listing/warn-by-default.md\n");
    for lint_name in to_link {
        let lint_def =
            lints.iter().find(|l| l.name == lint_name.replace("-", "_")).ok_or_else(|| {
                format!(
                    "`rustc -W help` defined lint `{}` but that lint does not appear to exist",
                    lint_name
                )
            })?;
        write!(
            result,
            "[{}]: listing/{}#{}\n",
            lint_name,
            lint_def.level.doc_filename(),
            lint_name
        )
        .unwrap();
    }
    Ok(result)
}
