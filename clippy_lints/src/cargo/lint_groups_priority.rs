use super::LINT_GROUPS_PRIORITY;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_lint::{unerased_lint_store, LateContext};
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

#[derive(Deserialize, Debug)]
struct Lints {
    #[serde(default)]
    rust: LintTable,
    #[serde(default)]
    clippy: LintTable,
}

#[derive(Deserialize, Debug)]
struct CargoToml {
    lints: Lints,
}

#[derive(Default, Debug)]
struct LintsAndGroups {
    lints: Vec<Spanned<String>>,
    groups: Vec<(Spanned<String>, Spanned<LintConfig>)>,
}

fn toml_span(range: Range<usize>, file: &SourceFile) -> Span {
    Span::new(
        file.start_pos + BytePos::from_usize(range.start),
        file.start_pos + BytePos::from_usize(range.end),
        SyntaxContext::root(),
        None,
    )
}

fn check_table(cx: &LateContext<'_>, table: LintTable, groups: &FxHashSet<&str>, file: &SourceFile) {
    let mut by_priority = BTreeMap::<_, LintsAndGroups>::new();
    for (name, config) in table {
        let lints_and_groups = by_priority.entry(config.as_ref().priority()).or_default();
        if groups.contains(name.get_ref().as_str()) {
            lints_and_groups.groups.push((name, config));
        } else {
            lints_and_groups.lints.push(name);
        }
    }
    let low_priority = by_priority
        .iter()
        .find(|(_, lints_and_groups)| !lints_and_groups.lints.is_empty())
        .map_or(-1, |(&lowest_lint_priority, _)| lowest_lint_priority - 1);

    for (priority, LintsAndGroups { lints, groups }) in by_priority {
        let Some(last_lint_alphabetically) = lints.last() else {
            continue;
        };

        for (group, config) in groups {
            span_lint_and_then(
                cx,
                LINT_GROUPS_PRIORITY,
                toml_span(group.span(), file),
                &format!(
                    "lint group `{}` has the same priority ({priority}) as a lint",
                    group.as_ref()
                ),
                |diag| {
                    let config_span = toml_span(config.span(), file);
                    if config.as_ref().is_implicit() {
                        diag.span_label(config_span, "has an implicit priority of 0");
                    }
                    // add the label to next lint after this group that has the same priority
                    let lint = lints
                        .iter()
                        .filter(|lint| lint.span().start > group.span().start)
                        .min_by_key(|lint| lint.span().start)
                        .unwrap_or(last_lint_alphabetically);
                    diag.span_label(toml_span(lint.span(), file), "has the same priority as this lint");
                    diag.note("the order of the lints in the table is ignored by Cargo");
                    let mut suggestion = String::new();
                    Serialize::serialize(
                        &LintConfigTable {
                            level: config.as_ref().level().into(),
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
    }
}
