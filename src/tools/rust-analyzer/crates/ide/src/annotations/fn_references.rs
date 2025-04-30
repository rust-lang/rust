//! This module implements a methods and free functions search in the specified file.
//! We have to skip tests, so cannot reuse file_structure module.

use hir::Semantics;
use ide_assists::utils::test_related_attribute_syn;
use ide_db::RootDatabase;
use syntax::{AstNode, SyntaxNode, TextRange, ast, ast::HasName};

use crate::FileId;

pub(super) fn find_all_methods(
    db: &RootDatabase,
    file_id: FileId,
) -> Vec<(TextRange, Option<TextRange>)> {
    let sema = Semantics::new(db);
    let source_file = sema.parse_guess_edition(file_id);
    source_file.syntax().descendants().filter_map(method_range).collect()
}

fn method_range(item: SyntaxNode) -> Option<(TextRange, Option<TextRange>)> {
    ast::Fn::cast(item).and_then(|fn_def| {
        if test_related_attribute_syn(&fn_def).is_some() {
            None
        } else {
            Some((
                fn_def.syntax().text_range(),
                fn_def.name().map(|name| name.syntax().text_range()),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use syntax::TextRange;

    use crate::TextSize;
    use crate::fixture;
    use std::ops::RangeInclusive;

    #[test]
    fn test_find_all_methods() {
        let (analysis, pos) = fixture::position(
            r#"
            fn private_fn() {$0}

            pub fn pub_fn() {}

            pub fn generic_fn<T>(arg: T) {}
        "#,
        );

        let refs = super::find_all_methods(&analysis.db, pos.file_id);
        check_result(&refs, &[3..=13, 27..=33, 47..=57]);
    }

    #[test]
    fn test_find_trait_methods() {
        let (analysis, pos) = fixture::position(
            r#"
            trait Foo {
                fn bar() {$0}
                fn baz() {}
            }
        "#,
        );

        let refs = super::find_all_methods(&analysis.db, pos.file_id);
        check_result(&refs, &[19..=22, 35..=38]);
    }

    #[test]
    fn test_skip_tests() {
        let (analysis, pos) = fixture::position(
            r#"
            //- /lib.rs
            #[test]
            fn foo() {$0}

            pub fn pub_fn() {}

            mod tests {
                #[test]
                fn bar() {}
            }
        "#,
        );

        let refs = super::find_all_methods(&analysis.db, pos.file_id);
        check_result(&refs, &[28..=34]);
    }

    fn check_result(refs: &[(TextRange, Option<TextRange>)], expected: &[RangeInclusive<u32>]) {
        assert_eq!(refs.len(), expected.len());

        for (i, &(full, focus)) in refs.iter().enumerate() {
            let range = &expected[i];
            let item = focus.unwrap_or(full);
            assert_eq!(TextSize::from(*range.start()), item.start());
            assert_eq!(TextSize::from(*range.end()), item.end());
        }
    }
}
