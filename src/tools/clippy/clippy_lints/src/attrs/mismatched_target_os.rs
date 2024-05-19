use super::{Attribute, MISMATCHED_TARGET_OS};
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::{MetaItemKind, NestedMetaItem};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;
use rustc_span::{sym, Span};

static UNIX_SYSTEMS: &[&str] = &[
    "android",
    "dragonfly",
    "emscripten",
    "freebsd",
    "fuchsia",
    "haiku",
    "illumos",
    "ios",
    "l4re",
    "linux",
    "macos",
    "netbsd",
    "openbsd",
    "redox",
    "solaris",
    "vxworks",
];

// NOTE: windows is excluded from the list because it's also a valid target family.
static NON_UNIX_SYSTEMS: &[&str] = &["hermit", "none", "wasi"];

pub(super) fn check(cx: &EarlyContext<'_>, attr: &Attribute) {
    fn find_os(name: &str) -> Option<&'static str> {
        UNIX_SYSTEMS
            .iter()
            .chain(NON_UNIX_SYSTEMS.iter())
            .find(|&&os| os == name)
            .copied()
    }

    fn is_unix(name: &str) -> bool {
        UNIX_SYSTEMS.iter().any(|&os| os == name)
    }

    fn find_mismatched_target_os(items: &[NestedMetaItem]) -> Vec<(&str, Span)> {
        let mut mismatched = Vec::new();

        for item in items {
            if let NestedMetaItem::MetaItem(meta) = item {
                match &meta.kind {
                    MetaItemKind::List(list) => {
                        mismatched.extend(find_mismatched_target_os(list));
                    },
                    MetaItemKind::Word => {
                        if let Some(ident) = meta.ident()
                            && let Some(os) = find_os(ident.name.as_str())
                        {
                            mismatched.push((os, ident.span));
                        }
                    },
                    MetaItemKind::NameValue(..) => {},
                }
            }
        }

        mismatched
    }

    if attr.has_name(sym::cfg)
        && let Some(list) = attr.meta_item_list()
        && let mismatched = find_mismatched_target_os(&list)
        && !mismatched.is_empty()
    {
        let mess = "operating system used in target family position";

        span_lint_and_then(cx, MISMATCHED_TARGET_OS, attr.span, mess, |diag| {
            // Avoid showing the unix suggestion multiple times in case
            // we have more than one mismatch for unix-like systems
            let mut unix_suggested = false;

            for (os, span) in mismatched {
                let sugg = format!("target_os = \"{os}\"");
                diag.span_suggestion(span, "try", sugg, Applicability::MaybeIncorrect);

                if !unix_suggested && is_unix(os) {
                    diag.help("did you mean `unix`?");
                    unix_suggested = true;
                }
            }
        });
    }
}
