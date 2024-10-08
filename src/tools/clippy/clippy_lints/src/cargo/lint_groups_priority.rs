use super::LINT_GROUPS_PRIORITY;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_lint::{LateContext, unerased_lint_store};
use rustc_span::{BytePos, Pos, SourceFile, Span, SyntaxContext};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;
use toml::Spanned;

#[derive(Deserialize, Serialize, Debug)]
struct LintConfigTable {
    level: String,
    priority: Option<i64>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum LintConfig {
    Level(String),
    Table(LintConfigTable),
}

impl LintConfig {
    fn level(&self) -> &str {
        match self {
            LintConfig::Level(level) => level,
            LintConfig::Table(table) => &table.level,
        }
    }

    fn priority(&self) -> i64 {
        match self {
            LintConfig::Level(_) => 0,
            LintConfig::Table(table) => table.priority.unwrap_or(0),
        }
    }

    fn is_implicit(&self) -> bool {
        if let LintConfig::Table(table) = self {
            table.priority.is_none()
        } else {
            true
        }
    }
}

type LintTable = BTreeMap<Spanned<String>, Spanned<LintConfig>>;

#[derive(Deserialize, Debug, Default)]
struct Lints {
    #[serde(default)]
    rust: LintTable,
    #[serde(default)]
    clippy: LintTable,
}

#[derive(Deserialize, Debug, Default)]
struct Workspace {
    #[serde(default)]
    lints: Lints,
}

#[derive(Deserialize, Debug)]
struct CargoToml {
    #[serde(default)]
    lints: Lints,
    #[serde(default)]
    workspace: Workspace,
}

fn toml_span(range: Range<usize>, file: &SourceFile) -> Span {
    Span::new(
        file.start_pos + BytePos::from_usize(range.start),
        file.start_pos + BytePos::from_usize(range.end),
        SyntaxContext::root(),
        None,
    )
}

fn check_table(cx: &LateContext<'_>, table: LintTable, known_groups: &FxHashSet<&str>, file: &SourceFile) {
    let mut lints = Vec::new();
    let mut groups = Vec::new();
    for (name, config) in table {
        if name.get_ref() == "warnings" {
            continue;
        }

        if known_groups.contains(name.get_ref().as_str()) {
            groups.push((name, config));
        } else {
            lints.push((name, config.into_inner()));
        }
    }

    for (group, group_config) in groups {
        let priority = group_config.get_ref().priority();
        let level = group_config.get_ref().level();
        if let Some((conflict, _)) = lints
            .iter()
            .rfind(|(_, lint_config)| lint_config.priority() == priority && lint_config.level() != level)
        {
            span_lint_and_then(
                cx,
                LINT_GROUPS_PRIORITY,
                toml_span(group.span(), file),
                format!(
                    "lint group `{}` has the same priority ({priority}) as a lint",
                    group.as_ref()
                ),
                |diag| {
                    let config_span = toml_span(group_config.span(), file);

                    if group_config.as_ref().is_implicit() {
                        diag.span_label(config_span, "has an implicit priority of 0");
                    }
                    diag.span_label(toml_span(conflict.span(), file), "has the same priority as this lint");
                    diag.note("the order of the lints in the table is ignored by Cargo");

                    let mut suggestion = String::new();
                    let low_priority = lints
                        .iter()
                        .map(|(_, config)| config.priority().saturating_sub(1))
                        .min()
                        .unwrap_or(-1);
                    Serialize::serialize(
                        &LintConfigTable {
                            level: level.into(),
                            priority: Some(low_priority),
                        },
                        toml::ser::ValueSerializer::new(&mut suggestion),
                    )
                    .unwrap();
                    diag.span_suggestion_verbose(
                        config_span,
                        format!(
                            "to have lints override the group set `{}` to a lower priority",
                            group.as_ref()
                        ),
                        suggestion,
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }
}

pub fn check(cx: &LateContext<'_>) {
    if let Ok(file) = cx.tcx.sess.source_map().load_file(Path::new("Cargo.toml"))
        && let Some(src) = file.src.as_deref()
        && let Ok(cargo_toml) = toml::from_str::<CargoToml>(src)
    {
        let mut rustc_groups = FxHashSet::default();
        let mut clippy_groups = FxHashSet::default();
        for (group, ..) in unerased_lint_store(cx.tcx.sess).get_lint_groups() {
            match group.split_once("::") {
                None => {
                    rustc_groups.insert(group);
                },
                Some(("clippy", group)) => {
                    clippy_groups.insert(group);
                },
                _ => {},
            }
        }

        check_table(cx, cargo_toml.lints.rust, &rustc_groups, &file);
        check_table(cx, cargo_toml.lints.clippy, &clippy_groups, &file);
        check_table(cx, cargo_toml.workspace.lints.rust, &rustc_groups, &file);
        check_table(cx, cargo_toml.workspace.lints.clippy, &clippy_groups, &file);
    }
}
