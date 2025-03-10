use hir::Semantics;
use ide_db::{FilePosition, RootDatabase};
use syntax::{
    algo::find_node_at_offset,
    ast::{self, AstNode},
};

use crate::NavigationTarget;

/// This returns `Vec` because a module may be included from several places.
pub(crate) fn children_modules(db: &RootDatabase, position: FilePosition) -> Vec<NavigationTarget> {
    let sema = Semantics::new(db);
    let source_file = sema.parse_guess_edition(position.file_id);
    // First go to the parent module which contains the cursor
    let module = find_node_at_offset::<ast::Module>(source_file.syntax(), position.offset);

    match module {
        Some(module) => {
            // Return all the children module inside the ItemList of the parent module
            sema.to_def(&module)
                .into_iter()
                .flat_map(|module| module.children(db))
                .map(|module| NavigationTarget::from_module_to_decl(db, module).call_site())
                .collect()
        }
        None => {
            // Return all the children module inside the source file
            sema.file_to_module_defs(position.file_id)
                .flat_map(|module| module.children(db))
                .map(|module| NavigationTarget::from_module_to_decl(db, module).call_site())
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use ide_db::FileRange;

    use crate::fixture;

    fn check_children_module(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);
        let navs = analysis.children_modules(position).unwrap();
        let navs = navs
            .iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .collect::<Vec<_>>();
        assert_eq!(expected.into_iter().map(|(fr, _)| fr).collect::<Vec<_>>(), navs);
    }

    #[test]
    fn test_resolve_children_module() {
        check_children_module(
            r#"
//- /lib.rs
$0
mod foo;
  //^^^

//- /foo.rs
// empty
"#,
        );
    }

    #[test]
    fn test_resolve_children_module_on_module_decl() {
        check_children_module(
            r#"
//- /lib.rs
mod $0foo;
//- /foo.rs
mod bar;
  //^^^

//- /foo/bar.rs
// empty
"#,
        );
    }

    #[test]
    fn test_resolve_children_module_for_inline() {
        check_children_module(
            r#"
//- /lib.rs
mod foo {
    mod $0bar {
        mod baz {}
    }     //^^^
}
"#,
        );
    }

    #[test]
    fn test_resolve_multi_child_module() {
        check_children_module(
            r#"
//- /main.rs
$0
mod foo;
  //^^^
mod bar;
  //^^^
  
//- /foo.rs
// empty

//- /bar.rs
// empty
"#,
        );
    }
}
