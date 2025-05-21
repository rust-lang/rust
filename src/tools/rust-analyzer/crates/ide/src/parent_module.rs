use hir::{Semantics, crate_def_map};
use ide_db::{
    FileId, FilePosition, RootDatabase,
    base_db::{Crate, RootQueryDb},
};
use itertools::Itertools;
use syntax::{
    algo::find_node_at_offset,
    ast::{self, AstNode},
};

use crate::NavigationTarget;

// Feature: Parent Module
//
// Navigates to the parent module of the current module.
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: Locate parent module** |
//
// ![Parent Module](https://user-images.githubusercontent.com/48062697/113065580-04c21800-91b1-11eb-9a32-00086161c0bd.gif)

/// This returns `Vec` because a module may be included from several places.
pub(crate) fn parent_module(db: &RootDatabase, position: FilePosition) -> Vec<NavigationTarget> {
    let sema = Semantics::new(db);
    let source_file = sema.parse_guess_edition(position.file_id);

    let mut module = find_node_at_offset::<ast::Module>(source_file.syntax(), position.offset);

    // If cursor is literally on `mod foo`, go to the grandpa.
    if let Some(m) = &module {
        if !m
            .item_list()
            .is_some_and(|it| it.syntax().text_range().contains_inclusive(position.offset))
        {
            cov_mark::hit!(test_resolve_parent_module_on_module_decl);
            module = m.syntax().ancestors().skip(1).find_map(ast::Module::cast);
        }
    }

    match module {
        Some(module) => sema
            .to_def(&module)
            .into_iter()
            .flat_map(|module| NavigationTarget::from_module_to_decl(db, module))
            .collect(),
        None => sema
            .file_to_module_defs(position.file_id)
            .flat_map(|module| NavigationTarget::from_module_to_decl(db, module))
            .collect(),
    }
}

/// This returns `Vec` because a module may be included from several places.
pub(crate) fn crates_for(db: &RootDatabase, file_id: FileId) -> Vec<Crate> {
    db.relevant_crates(file_id)
        .iter()
        .copied()
        .filter(|&crate_id| {
            crate_def_map(db, crate_id).modules_for_file(db, file_id).next().is_some()
        })
        .sorted()
        .collect()
}

#[cfg(test)]
mod tests {
    use ide_db::FileRange;

    use crate::fixture;

    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);
        let navs = analysis.parent_module(position).unwrap();
        let navs = navs
            .iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .collect::<Vec<_>>();
        assert_eq!(expected.into_iter().map(|(fr, _)| fr).collect::<Vec<_>>(), navs);
    }

    #[test]
    fn test_resolve_parent_module() {
        check(
            r#"
//- /lib.rs
mod foo;
  //^^^

//- /foo.rs
$0// empty
"#,
        );
    }

    #[test]
    fn test_resolve_parent_module_on_module_decl() {
        cov_mark::check!(test_resolve_parent_module_on_module_decl);
        check(
            r#"
//- /lib.rs
mod foo;
  //^^^
//- /foo.rs
mod $0bar;

//- /foo/bar.rs
// empty
"#,
        );
    }

    #[test]
    fn test_resolve_parent_module_for_inline() {
        check(
            r#"
//- /lib.rs
mod foo {
    mod bar {
        mod baz { $0 }
    }     //^^^
}
"#,
        );
    }

    #[test]
    fn test_resolve_multi_parent_module() {
        check(
            r#"
//- /main.rs
mod foo;
  //^^^
#[path = "foo.rs"]
mod bar;
  //^^^
//- /foo.rs
$0
"#,
        );
    }

    #[test]
    fn test_resolve_crate_root() {
        let (analysis, file_id) = fixture::file(
            r#"
//- /foo.rs
$0
//- /main.rs
mod foo;
"#,
        );
        assert_eq!(analysis.crates_for(file_id).unwrap().len(), 1);
    }

    #[test]
    fn test_resolve_multi_parent_crate() {
        let (analysis, file_id) = fixture::file(
            r#"
//- /baz.rs
$0
//- /foo.rs crate:foo
mod baz;
//- /bar.rs crate:bar
mod baz;
"#,
        );
        assert_eq!(analysis.crates_for(file_id).unwrap().len(), 2);
    }
}
