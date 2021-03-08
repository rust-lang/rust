use hir::Semantics;
use ide_db::base_db::{CrateId, FileId, FilePosition};
use ide_db::RootDatabase;
use syntax::{
    algo::find_node_at_offset,
    ast::{self, AstNode},
};

use crate::NavigationTarget;

// Feature: Parent Module
//
// Navigates to the parent module of the current module.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: Locate parent module**
// |===

/// This returns `Vec` because a module may be included from several places. We
/// don't handle this case yet though, so the Vec has length at most one.
pub(crate) fn parent_module(db: &RootDatabase, position: FilePosition) -> Vec<NavigationTarget> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);

    let mut module = find_node_at_offset::<ast::Module>(source_file.syntax(), position.offset);

    // If cursor is literally on `mod foo`, go to the grandpa.
    if let Some(m) = &module {
        if !m
            .item_list()
            .map_or(false, |it| it.syntax().text_range().contains_inclusive(position.offset))
        {
            cov_mark::hit!(test_resolve_parent_module_on_module_decl);
            module = m.syntax().ancestors().skip(1).find_map(ast::Module::cast);
        }
    }

    let module = match module {
        Some(module) => sema.to_def(&module),
        None => sema.to_module_def(position.file_id),
    };
    let module = match module {
        None => return Vec::new(),
        Some(it) => it,
    };
    let nav = NavigationTarget::from_module_to_decl(db, module);
    vec![nav]
}

/// Returns `Vec` for the same reason as `parent_module`
pub(crate) fn crate_for(db: &RootDatabase, file_id: FileId) -> Vec<CrateId> {
    let sema = Semantics::new(db);
    let module = match sema.to_module_def(file_id) {
        Some(it) => it,
        None => return Vec::new(),
    };
    let krate = module.krate();
    vec![krate.into()]
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::FileRange;

    use crate::fixture;

    fn check(ra_fixture: &str) {
        let (analysis, position, expected) = fixture::nav_target_annotation(ra_fixture);
        let mut navs = analysis.parent_module(position).unwrap();
        assert_eq!(navs.len(), 1);
        let nav = navs.pop().unwrap();
        assert_eq!(expected, FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() });
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
    fn test_resolve_crate_root() {
        let (analysis, file_id) = fixture::file(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
$0
"#,
        );
        assert_eq!(analysis.crate_for(file_id).unwrap().len(), 1);
    }
}
