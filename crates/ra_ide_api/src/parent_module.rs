use ra_db::{FilePosition, FileId, CrateId};

use crate::{NavigationTarget, db::RootDatabase};

/// This returns `Vec` because a module may be included from several places. We
/// don't handle this case yet though, so the Vec has length at most one.
pub(crate) fn parent_module(db: &RootDatabase, position: FilePosition) -> Vec<NavigationTarget> {
    let module = match hir::source_binder::module_from_position(db, position) {
        None => return Vec::new(),
        Some(it) => it,
    };
    let nav = NavigationTarget::from_module_to_decl(db, module);
    vec![nav]
}

/// Returns `Vec` for the same reason as `parent_module`
pub(crate) fn crate_for(db: &RootDatabase, file_id: FileId) -> Vec<CrateId> {
    let module = match hir::source_binder::module_from_file_id(db, file_id) {
        Some(it) => it,
        None => return Vec::new(),
    };
    let krate = match module.krate(db) {
        Some(it) => it,
        None => return Vec::new(),
    };
    vec![krate.crate_id()]
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::analysis_and_position;

    #[test]
    fn test_resolve_parent_module() {
        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            mod foo;
            //- /foo.rs
            <|>// empty
            ",
        );
        let nav = analysis.parent_module(pos).unwrap().pop().unwrap();
        nav.assert_match("foo MODULE FileId(1) [0; 8)");
    }

    #[test]
    fn test_resolve_parent_module_for_inline() {
        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            mod foo {
                mod bar {
                    mod baz { <|> }
                }
            }
            ",
        );
        let nav = analysis.parent_module(pos).unwrap().pop().unwrap();
        nav.assert_match("baz MODULE FileId(1) [32; 44)");
    }
}
