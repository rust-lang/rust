//! This module implements a methods and free functions search in the specified file.
//! We have to skip tests, so cannot reuse file_structure module.

use hir::Semantics;
use ide_assists::utils::test_related_attribute;
use ide_db::RootDatabase;
use syntax::{ast, ast::HasName, AstNode, SyntaxNode};

use crate::{FileId, FileRange};

pub(crate) fn find_all_methods(db: &RootDatabase, file_id: FileId) -> Vec<FileRange> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(file_id);
    source_file.syntax().descendants().filter_map(|it| method_range(it, file_id)).collect()
}

fn method_range(item: SyntaxNode, file_id: FileId) -> Option<FileRange> {
    ast::Fn::cast(item).and_then(|fn_def| {
        if test_related_attribute(&fn_def).is_some() {
            None
        } else {
            fn_def.name().map(|name| FileRange { file_id, range: name.syntax().text_range() })
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::fixture;
    use crate::{FileRange, TextSize};
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

        let refs = analysis.find_all_methods(pos.file_id).unwrap();
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

        let refs = analysis.find_all_methods(pos.file_id).unwrap();
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

        let refs = analysis.find_all_methods(pos.file_id).unwrap();
        check_result(&refs, &[28..=34]);
    }

    fn check_result(refs: &[FileRange], expected: &[RangeInclusive<u32>]) {
        assert_eq!(refs.len(), expected.len());

        for (i, item) in refs.iter().enumerate() {
            let range = &expected[i];
            assert_eq!(TextSize::from(*range.start()), item.range.start());
            assert_eq!(TextSize::from(*range.end()), item.range.end());
        }
    }
}
