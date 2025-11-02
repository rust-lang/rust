use super::LINT_GROUPS_PRIORITY;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_lint::{LateContext, unerased_lint_store};
use rustc_span::{BytePos, Pos, SourceFile, Span, SyntaxContext};
use std::ops::Range;
use std::path::Path;
use toml::Spanned;
use toml::de::{DeTable, DeValue};

fn toml_span(range: Range<usize>, file: &SourceFile) -> Span {
    Span::new(
        file.start_pos + BytePos::from_usize(range.start),
        file.start_pos + BytePos::from_usize(range.end),
        SyntaxContext::root(),
        None,
    )
}

struct LintConfig<'a> {
    sp: Range<usize>,
    level: &'a str,
    priority: Option<i64>,
}
impl<'a> LintConfig<'a> {
    fn priority(&self) -> i64 {
        self.priority.unwrap_or(0)
    }

    fn is_implicit(&self) -> bool {
        self.priority.is_none()
    }

    fn parse(value: &'a Spanned<DeValue<'a>>) -> Option<Self> {
        let sp = value.span();
        let (level, priority) = match value.get_ref() {
            DeValue::String(level) => (&**level, None),
            DeValue::Table(tbl) => {
                let level = tbl.get("level")?.get_ref().as_str()?;
                let priority = if let Some(priority) = tbl.get("priority") {
                    let priority = priority.get_ref().as_integer()?;
                    Some(i64::from_str_radix(priority.as_str(), priority.radix()).ok()?)
                } else {
                    None
                };
                (level, priority)
            },
            _ => return None,
        };
        Some(Self { sp, level, priority })
    }
}

fn check_table(cx: &LateContext<'_>, table: &DeTable<'_>, known_groups: &FxHashSet<&str>, file: &SourceFile) {
    let mut lints = Vec::new();
    let mut groups = Vec::new();
    for (name, config) in table {
        if name.get_ref() != "warnings"
            && let Some(config) = LintConfig::parse(config)
        {
            if known_groups.contains(&**name.get_ref()) {
                groups.push((name, config));
            } else {
                lints.push((name, config));
            }
        }
    }

    for (group, group_config) in groups {
        if let Some((conflict, _)) = lints.iter().rfind(|(_, lint_config)| {
            lint_config.priority() == group_config.priority() && lint_config.level != group_config.level
        }) {
            span_lint_and_then(
                cx,
                LINT_GROUPS_PRIORITY,
                toml_span(group.span(), file),
                format!(
                    "lint group `{}` has the same priority ({}) as a lint",
                    group.as_ref(),
                    group_config.priority(),
                ),
                |diag| {
                    let config_span = toml_span(group_config.sp.clone(), file);

                    if group_config.is_implicit() {
                        diag.span_label(config_span, "has an implicit priority of 0");
                    }
                    diag.span_label(toml_span(conflict.span(), file), "has the same priority as this lint");
                    diag.note("the order of the lints in the table is ignored by Cargo");

                    let low_priority = lints
                        .iter()
                        .map(|(_, lint_config)| lint_config.priority().saturating_sub(1))
                        .min()
                        .unwrap_or(-1);
                    diag.span_suggestion_verbose(
                        config_span,
                        format!(
                            "to have lints override the group set `{}` to a lower priority",
                            group.as_ref()
                        ),
                        format!("{{ level = {:?}, priority = {low_priority} }}", group_config.level,),
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }
}

struct LintTbls<'a> {
    rust: Option<&'a DeTable<'a>>,
    clippy: Option<&'a DeTable<'a>>,
}
fn get_lint_tbls<'a>(tbl: &'a DeTable<'a>) -> LintTbls<'a> {
    if let Some(lints) = tbl.get("lints")
        && let Some(lints) = lints.get_ref().as_table()
    {
        let rust = lints.get("rust").and_then(|x| x.get_ref().as_table());
        let clippy = lints.get("clippy").and_then(|x| x.get_ref().as_table());
        LintTbls { rust, clippy }
    } else {
        LintTbls {
            rust: None,
            clippy: None,
        }
    }
}

pub fn check(cx: &LateContext<'_>) {
    if let Ok(file) = cx.tcx.sess.source_map().load_file(Path::new("Cargo.toml"))
        && let Some(src) = file.src.as_deref()
        && let Ok(cargo_toml) = DeTable::parse(src)
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

        let lints = get_lint_tbls(cargo_toml.get_ref());
        if let Some(lints) = lints.rust {
            check_table(cx, lints, &rustc_groups, &file);
        }
        if let Some(lints) = lints.clippy {
            check_table(cx, lints, &clippy_groups, &file);
        }
        if let Some(tbl) = cargo_toml.get_ref().get("workspace")
            && let Some(tbl) = tbl.get_ref().as_table()
        {
            let lints = get_lint_tbls(tbl);
            if let Some(lints) = lints.rust {
                check_table(cx, lints, &rustc_groups, &file);
            }
            if let Some(lints) = lints.clippy {
                check_table(cx, lints, &clippy_groups, &file);
            }
        }
    }
}
