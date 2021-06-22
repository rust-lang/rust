use hir::Semantics;
use ide_db::{
    base_db::FilePosition,
    defs::Definition,
    search::{FileReference, ReferenceAccess, SearchScope},
    RootDatabase,
};
use syntax::{AstNode, TextRange};

use crate::{display::TryToNav, references, NavigationTarget};

pub struct DocumentHighlight {
    pub range: TextRange,
    pub access: Option<ReferenceAccess>,
}

// Feature: Document highlight
//
// Highlights the definition and its all references of the item at the cursor location in the current file.
pub(crate) fn document_highlight(
    sema: &Semantics<RootDatabase>,
    position: FilePosition,
) -> Option<Vec<DocumentHighlight>> {
    let _p = profile::span("document_highlight");
    let syntax = sema.parse(position.file_id).syntax().clone();
    let def = references::find_def(sema, &syntax, position)?;
    let usages = def.usages(sema).set_scope(Some(SearchScope::single_file(position.file_id))).all();

    let declaration = match def {
        Definition::ModuleDef(hir::ModuleDef::Module(module)) => {
            Some(NavigationTarget::from_module_to_decl(sema.db, module))
        }
        def => def.try_to_nav(sema.db),
    }
    .filter(|decl| decl.file_id == position.file_id)
    .and_then(|decl| {
        let range = decl.focus_range?;
        let access = references::decl_access(&def, &syntax, range);
        Some(DocumentHighlight { range, access })
    });

    let file_refs = usages.references.get(&position.file_id).map_or(&[][..], Vec::as_slice);
    let mut res = Vec::with_capacity(file_refs.len() + 1);
    res.extend(declaration);
    res.extend(
        file_refs
            .iter()
            .map(|&FileReference { access, range, .. }| DocumentHighlight { range, access }),
    );
    Some(res)
}

#[cfg(test)]
mod tests {
    use crate::fixture;

    use super::*;

    fn check(ra_fixture: &str) {
        let (analysis, pos, annotations) = fixture::annotations(ra_fixture);
        let hls = analysis.highlight_document(pos).unwrap().unwrap();

        let mut expected = annotations
            .into_iter()
            .map(|(r, access)| (r.range, (!access.is_empty()).then(|| access)))
            .collect::<Vec<_>>();

        let mut actual = hls
            .into_iter()
            .map(|hl| {
                (
                    hl.range,
                    hl.access.map(|it| {
                        match it {
                            ReferenceAccess::Read => "read",
                            ReferenceAccess::Write => "write",
                        }
                        .to_string()
                    }),
                )
            })
            .collect::<Vec<_>>();
        actual.sort_by_key(|(range, _)| range.start());
        expected.sort_by_key(|(range, _)| range.start());

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_hl_module() {
        check(
            r#"
//- /lib.rs
mod foo$0;
 // ^^^
//- /foo.rs
struct Foo;
"#,
        );
    }

    #[test]
    fn test_hl_self_in_crate_root() {
        check(
            r#"
//- /lib.rs
use self$0;
"#,
        );
    }

    #[test]
    fn test_hl_self_in_module() {
        check(
            r#"
//- /lib.rs
mod foo;
//- /foo.rs
use self$0;
"#,
        );
    }

    #[test]
    fn test_hl_local() {
        check(
            r#"
//- /lib.rs
fn foo() {
    let mut bar = 3;
         // ^^^ write
    bar$0;
 // ^^^ read
}
"#,
        );
    }
}
