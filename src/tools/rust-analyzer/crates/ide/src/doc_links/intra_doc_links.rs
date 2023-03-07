//! Helper tools for intra doc links.

const TYPES: ([&str; 9], [&str; 0]) =
    (["type", "struct", "enum", "mod", "trait", "union", "module", "prim", "primitive"], []);
const VALUES: ([&str; 8], [&str; 1]) =
    (["value", "function", "fn", "method", "const", "static", "mod", "module"], ["()"]);
const MACROS: ([&str; 2], [&str; 1]) = (["macro", "derive"], ["!"]);

/// Extract the specified namespace from an intra-doc-link if one exists.
///
/// # Examples
///
/// * `struct MyStruct` -> ("MyStruct", `Namespace::Types`)
/// * `panic!` -> ("panic", `Namespace::Macros`)
/// * `fn@from_intra_spec` -> ("from_intra_spec", `Namespace::Values`)
pub(super) fn parse_intra_doc_link(s: &str) -> (&str, Option<hir::Namespace>) {
    let s = s.trim_matches('`');

    [
        (hir::Namespace::Types, (TYPES.0.iter(), TYPES.1.iter())),
        (hir::Namespace::Values, (VALUES.0.iter(), VALUES.1.iter())),
        (hir::Namespace::Macros, (MACROS.0.iter(), MACROS.1.iter())),
    ]
    .into_iter()
    .find_map(|(ns, (mut prefixes, mut suffixes))| {
        if let Some(prefix) = prefixes.find(|&&prefix| {
            s.starts_with(prefix)
                && s.chars().nth(prefix.len()).map_or(false, |c| c == '@' || c == ' ')
        }) {
            Some((&s[prefix.len() + 1..], ns))
        } else {
            suffixes.find_map(|&suffix| s.strip_suffix(suffix).zip(Some(ns)))
        }
    })
    .map_or((s, None), |(s, ns)| (s, Some(ns)))
}

pub(super) fn strip_prefixes_suffixes(s: &str) -> &str {
    [
        (TYPES.0.iter(), TYPES.1.iter()),
        (VALUES.0.iter(), VALUES.1.iter()),
        (MACROS.0.iter(), MACROS.1.iter()),
    ]
    .into_iter()
    .find_map(|(mut prefixes, mut suffixes)| {
        if let Some(prefix) = prefixes.find(|&&prefix| {
            s.starts_with(prefix)
                && s.chars().nth(prefix.len()).map_or(false, |c| c == '@' || c == ' ')
        }) {
            Some(&s[prefix.len() + 1..])
        } else {
            suffixes.find_map(|&suffix| s.strip_suffix(suffix))
        }
    })
    .unwrap_or(s)
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use super::*;

    fn check(link: &str, expected: Expect) {
        let (l, a) = parse_intra_doc_link(link);
        let a = a.map_or_else(String::new, |a| format!(" ({a:?})"));
        expected.assert_eq(&format!("{l}{a}"));
    }

    #[test]
    fn test_name() {
        check("foo", expect![[r#"foo"#]]);
        check("struct Struct", expect![[r#"Struct (Types)"#]]);
        check("makro!", expect![[r#"makro (Macros)"#]]);
        check("fn@function", expect![[r#"function (Values)"#]]);
    }
}
