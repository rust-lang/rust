use hir::Semantics;
use ra_db::{CrateId, FileId, FilePosition};
use ra_ide_db::RootDatabase;
use ra_syntax::{
    algo::find_node_at_offset,
    ast::{self, AstNode},
};
use test_utils::mark;

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
            mark::hit!(test_resolve_parent_module_on_module_decl);
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
    use ra_cfg::CfgOptions;
    use ra_db::Env;
    use test_utils::mark;

    use crate::{
        mock_analysis::{analysis_and_position, MockAnalysis},
        AnalysisChange, CrateGraph,
        Edition::Edition2018,
    };

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
        nav.assert_match("foo MODULE FileId(1) 0..8");
    }

    #[test]
    fn test_resolve_parent_module_on_module_decl() {
        mark::check!(test_resolve_parent_module_on_module_decl);
        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            mod foo;

            //- /foo.rs
            mod <|>bar;

            //- /foo/bar.rs
            // empty
            ",
        );
        let nav = analysis.parent_module(pos).unwrap().pop().unwrap();
        nav.assert_match("foo MODULE FileId(1) 0..8");
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
        nav.assert_match("baz MODULE FileId(1) 32..44");
    }

    #[test]
    fn test_resolve_crate_root() {
        let mock = MockAnalysis::with_files(
            r#"
//- /bar.rs
mod foo;
//- /foo.rs
// empty
"#,
        );
        let root_file = mock.id_of("/bar.rs");
        let mod_file = mock.id_of("/foo.rs");
        let mut host = mock.analysis_host();
        assert!(host.analysis().crate_for(mod_file).unwrap().is_empty());

        let mut crate_graph = CrateGraph::default();
        let crate_id = crate_graph.add_crate_root(
            root_file,
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        let mut change = AnalysisChange::new();
        change.set_crate_graph(crate_graph);
        host.apply_change(change);

        assert_eq!(host.analysis().crate_for(mod_file).unwrap(), vec![crate_id]);
    }
}
