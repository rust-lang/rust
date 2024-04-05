use cargo_metadata::Metadata;
use clippy_utils::diagnostics::span_lint_and_help;
use rustc_lint::LateContext;
use rustc_span::DUMMY_SP;

use super::{NEGATIVE_FEATURE_NAMES, REDUNDANT_FEATURE_NAMES};

static PREFIXES: [&str; 8] = ["no-", "no_", "not-", "not_", "use-", "use_", "with-", "with_"];
static SUFFIXES: [&str; 2] = ["-support", "_support"];

pub(super) fn check(cx: &LateContext<'_>, metadata: &Metadata) {
    for package in &metadata.packages {
        let mut features: Vec<&String> = package.features.keys().collect();
        features.sort();
        for feature in features {
            let prefix_opt = {
                let i = PREFIXES.partition_point(|prefix| prefix < &feature.as_str());
                if i > 0 && feature.starts_with(PREFIXES[i - 1]) {
                    Some(PREFIXES[i - 1])
                } else {
                    None
                }
            };
            if let Some(prefix) = prefix_opt {
                lint(cx, feature, prefix, true);
            }

            let suffix_opt: Option<&str> = {
                let i = SUFFIXES.partition_point(|suffix| {
                    suffix.bytes().rev().cmp(feature.bytes().rev()) == std::cmp::Ordering::Less
                });
                if i > 0 && feature.ends_with(SUFFIXES[i - 1]) {
                    Some(SUFFIXES[i - 1])
                } else {
                    None
                }
            };
            if let Some(suffix) = suffix_opt {
                lint(cx, feature, suffix, false);
            }
        }
    }
}

fn is_negative_prefix(s: &str) -> bool {
    s.starts_with("no")
}

fn lint(cx: &LateContext<'_>, feature: &str, substring: &str, is_prefix: bool) {
    let is_negative = is_prefix && is_negative_prefix(substring);
    span_lint_and_help(
        cx,
        if is_negative {
            NEGATIVE_FEATURE_NAMES
        } else {
            REDUNDANT_FEATURE_NAMES
        },
        DUMMY_SP,
        format!(
            "the \"{substring}\" {} in the feature name \"{feature}\" is {}",
            if is_prefix { "prefix" } else { "suffix" },
            if is_negative { "negative" } else { "redundant" }
        ),
        None,
        format!(
            "consider renaming the feature to \"{}\"{}",
            if is_prefix {
                feature.strip_prefix(substring)
            } else {
                feature.strip_suffix(substring)
            }
            .unwrap(),
            if is_negative {
                ", but make sure the feature adds functionality"
            } else {
                ""
            }
        ),
    );
}

#[test]
fn test_prefixes_sorted() {
    let mut sorted_prefixes = PREFIXES;
    sorted_prefixes.sort_unstable();
    assert_eq!(PREFIXES, sorted_prefixes);
    let mut sorted_suffixes = SUFFIXES;
    sorted_suffixes.sort_by(|a, b| a.bytes().rev().cmp(b.bytes().rev()));
    assert_eq!(SUFFIXES, sorted_suffixes);
}
