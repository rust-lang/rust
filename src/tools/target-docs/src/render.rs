use eyre::{Context, OptionExt, Result};
use std::{fs, path::Path};

use crate::TargetInfo;

/// Renders a single target markdown file from the information obtained.
pub fn render_target_md(target: &TargetInfo) -> String {
    let render_header_option_bool = |bool| match bool {
        Some(true) => "Yes",
        Some(false) => "No",
        None => "?",
    };

    let mut doc = format!(
        "# {}\n\n**Tier: {}**\n\n**std: {}**\n\n**host tools: {}**\n\n",
        target.name,
        match target.metadata.tier {
            Some(1) => "1",
            Some(2) => "2",
            Some(3) => "3",
            _ => "UNKNOWN",
        },
        render_header_option_bool(target.metadata.std),
        render_header_option_bool(target.metadata.host_tools),
    );

    let mut section = |name: &str, content: &str| {
        doc.push_str("## ");
        doc.push_str(name.trim());
        doc.push('\n');
        doc.push_str(content.trim());
        doc.push_str("\n\n");
    };

    let maintainers_content = if target.maintainers.is_empty() {
        "This target does not have any maintainers!".to_owned()
    } else {
        format!(
            "This target is maintained by:\n{}",
            target
                .maintainers
                .iter()
                .map(|maintainer| {
                    let maintainer = if maintainer.starts_with('@') && !maintainer.contains(' ') {
                        format!(
                            "[@{0}](https://github.com/{0})",
                            maintainer.strip_prefix("@").unwrap()
                        )
                    } else {
                        maintainer.to_owned()
                    };

                    format!("- {maintainer}")
                })
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    section("Maintainers", &maintainers_content);

    for section_name in crate::SECTIONS {
        let value = target.sections.iter().find(|(name, _)| name == section_name);

        let section_content = match value {
            Some((_, value)) => value.clone(),
            None => "Unknown.".to_owned(),
        };
        section(section_name, &section_content);
    }

    let cfg_text = target
        .target_cfgs
        .iter()
        .map(|(key, value)| format!("- `{key}` = `{value}`"))
        .collect::<Vec<_>>()
        .join("\n");
    let cfg_content =
        format!("This target defines the following target-specific cfg values:\n{cfg_text}\n");

    section("cfg", &cfg_content);

    doc
}

/// Replaces inner part of the form
/// `<!-- {section_name} SECTION START --><!-- {section_name} SECTION END -->`
/// with replacement`.
fn replace_section(prev_content: &str, section_name: &str, replacement: &str) -> Result<String> {
    let magic_summary_start = format!("<!-- {section_name} SECTION START -->");
    let magic_summary_end = format!("<!-- {section_name} SECTION END -->");

    let (pre_target, target_and_after) = prev_content
        .split_once(&magic_summary_start)
        .ok_or_eyre("<!-- TARGET SECTION START --> not found")?;

    let (_, post_target) = target_and_after
        .split_once(&magic_summary_end)
        .ok_or_eyre("<!-- TARGET SECTION START --> not found")?;

    let new = format!("{pre_target}{replacement}{post_target}");
    Ok(new)
}

/// Renders the non-target files like `SUMMARY.md` that depend on the target.
pub fn render_static(check_only: bool, src_output: &Path, targets: &[TargetInfo]) -> Result<()> {
    let targets_file = src_output.join("platform-support").join("targets.md");
    let old_targets = fs::read_to_string(&targets_file).wrap_err("reading summary file")?;

    let target_list = targets
        .iter()
        .map(|target| format!("- [{0}](platform-support/targets/{0}.md)", target.name))
        .collect::<Vec<_>>()
        .join("\n");

    let new_targets =
        replace_section(&old_targets, "TARGET", &target_list).wrap_err("replacing targets.md")?;

    if !check_only {
        fs::write(targets_file, new_targets).wrap_err("writing targets.md")?;
    }

    let platform_support_main = src_output.join("platform-support.md");
    let platform_support_main_old =
        fs::read_to_string(&platform_support_main).wrap_err("reading platform-support.md")?;
    let platform_support_main_new =
        render_platform_support_tables(&platform_support_main_old, targets)?;

    if !check_only {
        fs::write(platform_support_main, platform_support_main_new)
            .wrap_err("writing platform-support.md")?;
    }

    let summary = src_output.join("SUMMARY.md");
    let summary_old = fs::read_to_string(&summary).wrap_err("reading SUMMARY.md")?;
    // indent the list
    let summary_new =
        replace_section(&summary_old, "TARGET_LIST", &target_list.replace("- ", "      - "))
            .wrap_err("replacig SUMMARY.md")?;
    if !check_only {
        fs::write(summary, summary_new).wrap_err("writing SUMAMRY.md")?;
    }

    Ok(())
}

impl TargetInfo {
    fn has_host_tools(&self) -> bool {
        self.metadata.host_tools.unwrap_or(false)
    }
}

fn render_platform_support_tables(content: &str, targets: &[TargetInfo]) -> Result<String> {
    let replace_table = |content, name, tier_table| -> Result<String> {
        let section_string = render_table(targets, tier_table)?;
        replace_section(content, name, &section_string).wrap_err("replacing platform support.md")
    };

    let content = replace_table(
        content,
        "TIER1HOST",
        TierTable {
            filter: |target| target.metadata.tier == Some(1),
            include_host: false,
            include_std: false,
        },
    )?;
    let content = replace_table(
        &content,
        "TIER2HOST",
        TierTable {
            filter: |target| target.metadata.tier == Some(2) && target.has_host_tools(),
            include_host: false,
            include_std: false,
        },
    )?;
    let content = replace_table(
        &content,
        "TIER2",
        TierTable {
            filter: |target| target.metadata.tier == Some(2) && !target.has_host_tools(),
            include_host: false,
            include_std: true,
        },
    )?;
    let content = replace_table(
        &content,
        "TIER3",
        TierTable {
            filter: |target| target.metadata.tier == Some(3),
            include_host: true,
            include_std: true,
        },
    )?;

    Ok(content)
}

fn render_table_option_bool(bool: Option<bool>) -> &'static str {
    match bool {
        Some(true) => "✓",
        Some(false) => " ",
        None => "?",
    }
}

struct TierTable {
    filter: fn(&TargetInfo) -> bool,
    include_std: bool,
    include_host: bool,
}

fn render_table(targets: &[TargetInfo], table: TierTable) -> Result<String> {
    let mut rows = Vec::new();

    let targets = targets.iter().filter(|target| (table.filter)(target));

    for target in targets {
        let meta = &target.metadata;

        let mut notes = meta.description.as_deref().unwrap_or("unknown").to_owned();

        if !target.footnotes.is_empty() {
            let footnotes_str = target
                .footnotes
                .iter()
                .map(|footnote| format!("[^{}]", footnote))
                .collect::<Vec<_>>()
                .join(" ");

            notes = format!("{notes} {footnotes_str}");
        }

        let std = if table.include_std {
            let std = render_table_option_bool(meta.std);
            format!(" | {std}")
        } else {
            String::new()
        };

        let host = if table.include_host {
            let host = render_table_option_bool(meta.host_tools);
            format!(" | {host}")
        } else {
            String::new()
        };

        rows.push(format!(
            "[`{0}`](platform-support/targets/{0}.md){std}{host} | {notes}",
            target.name
        ));
    }

    let result = rows.join("\n");

    Ok(result)
}
