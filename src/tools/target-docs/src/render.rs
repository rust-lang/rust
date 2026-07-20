use std::fmt::Write;
use std::fs;
use std::path::Path;

use eyre::{Context, Result, eyre};

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
        .ok_or_else(|| eyre!("{magic_summary_start} not found"))?;

    let (_, post_target) = target_and_after
        .split_once(&magic_summary_end)
        .ok_or_else(|| eyre!("{magic_summary_end} not found"))?;

    //let new = format!("{pre_target}{magic_summary_start}\n<!-- See `src/tools/target-docs` -->\n{replacement}\n{magic_summary_end}{post_target}");
    let new = format!("{pre_target}{replacement}{post_target}");
    Ok(new)
}

fn replace_file<P: AsRef<Path>, F: FnOnce(&str) -> Result<String>>(
    check_only: bool,
    src_output: &Path,
    file: P,
    replace_fn: F,
) -> Result<()> {
    let rel_path = file.as_ref();
    let filepath = src_output.join(rel_path);

    let old_content =
        fs::read_to_string(&filepath).wrap_err_with(|| eyre!("reading {:?}", rel_path))?;
    let new_content =
        replace_fn(&old_content).wrap_err_with(|| eyre!("replacing {:?}", rel_path))?;

    /*if new_content != old_content {
        eprintln!("{}: out of sync");
    }*/

    if !check_only {
        fs::write(filepath, new_content).wrap_err_with(|| eyre!("writing {:?}", rel_path))?;
    }

    Ok(())
}

/// Renders the non-target files like `SUMMARY.md` that depend on the target.
pub fn render_static(check_only: bool, src_output: &Path, targets: &[TargetInfo]) -> Result<()> {
    let target_list = targets
        .iter()
        .map(|target| format!("- [{0}](platform-support/targets/{0}.md)", target.name))
        .collect::<Vec<_>>()
        .join("\n");

    replace_file(check_only, src_output, "platform-support/targets.md", |content| {
        replace_section(content, "TARGET_LIST", &target_list)
    })?;

    let unmaintained_target_list = targets
        .iter()
        .filter(|target| target.maintainers.is_empty())
        .map(|target| format!("- [{0}](platform-support/targets/{0}.md)", target.name))
        .collect::<Vec<_>>()
        .join("\n");
    replace_file(check_only, src_output, "platform-support/unmaintained-targets.md", |content| {
        replace_section(content, "TARGET_LIST", &unmaintained_target_list)
    })?;

    replace_file(check_only, src_output, "platform-support.md", |content| {
        render_platform_support_tables(content, targets)
    })?;

    replace_file(check_only, src_output, "SUMMARY.md", |content| {
        // indent the list
        replace_section(content, "TARGET_LIST", &target_list.replace("- ", "      - "))
    })?;

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
        replace_section(content, name, &section_string).wrap_err("replacing platform-support.md")
    };

    let content = replace_table(
        content,
        "TIER1HOST",
        TierTable {
            filter: |target| target.metadata.tier == Some(1) && target.has_host_tools(),
            include_host: false,
            include_std: false,
            empty_msg: None,
        },
    )?;
    let content = replace_table(
        &content,
        "TIER1",
        TierTable {
            filter: |target| target.metadata.tier == Some(1) && !target.has_host_tools(),
            include_host: false,
            include_std: true,
            empty_msg: Some(
                "At this time, all Tier 1 targets are [Tier 1 with Host Tools](#tier-1-with-host-tools).",
            ),
        },
    )?;
    let content = replace_table(
        &content,
        "TIER2HOST",
        TierTable {
            filter: |target| target.metadata.tier == Some(2) && target.has_host_tools(),
            include_host: false,
            include_std: false,
            empty_msg: None,
        },
    )?;
    let content = replace_table(
        &content,
        "TIER2",
        TierTable {
            filter: |target| target.metadata.tier == Some(2) && !target.has_host_tools(),
            include_host: false,
            include_std: true,
            empty_msg: None,
        },
    )?;
    let content = replace_table(
        &content,
        "TIER3",
        TierTable {
            filter: |target| target.metadata.tier == Some(3),
            include_host: true,
            include_std: true,
            empty_msg: None,
        },
    )?;

    Ok(content)
}

fn render_table_option_bool(bool: Option<bool>) -> &'static str {
    match bool {
        Some(true) => "✓",
        Some(false) => "*",
        None => "?",
    }
}

struct TierTable {
    filter: fn(&TargetInfo) -> bool,
    include_std: bool,
    include_host: bool,
    empty_msg: Option<&'static str>,
}

fn render_table(targets: &[TargetInfo], table: TierTable) -> Result<String> {
    let mut count = 0;
    let mut out = String::with_capacity(1024);

    let mut row2 = String::with_capacity(32);

    out.push_str("target");
    row2.push_str("-------");

    if table.include_std {
        out.push_str(" | std");
        row2.push_str("|:---:");
    }

    if table.include_host {
        out.push_str(" | host");
        row2.push_str("|:----:");
    }

    out.push_str(" | notes\n");
    row2.push_str("|-------");
    out.push_str(&row2);
    drop(row2);

    for target in targets.iter().filter(|target| (table.filter)(target)) {
        count += 1;
        let meta = &target.metadata;

        write!(out, "\n[`{0}`](platform-support/targets/{0}.md)", target.name)?;

        if table.include_std {
            write!(out, " | {}", render_table_option_bool(meta.std))?;
        }

        if table.include_host {
            write!(out, " | {}", render_table_option_bool(meta.host_tools))?;
        }

        write!(out, " | {}", meta.description.as_deref().unwrap_or_default())?;

        for footnote in &target.footnotes {
            write!(out, " [^{}]", footnote)?;
        }
    }

    if count == 0
        && let Some(msg) = table.empty_msg
    {
        out.clear();
        out.push_str(msg);
    }

    Ok(out)
}
