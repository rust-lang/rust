//! FIXME: write short doc here

use ra_db::{CrateId, FileId, FilePosition, SourceDatabase};

use crate::{db::RootDatabase, NavigationTarget};

/// This returns `Vec` because a module may be included from several places. We
/// don't handle this case yet though, so the Vec has length at most one.
pub(crate) fn parent_module(db: &RootDatabase, position: FilePosition) -> Vec<NavigationTarget> {
    let src = hir::ModuleSource::from_position(db, position);
    let module = match hir::Module::from_definition(
        db,
        hir::InFile { file_id: position.file_id.into(), value: src },
    ) {
        None => return Vec::new(),
        Some(it) => it,
    };
    let nav = NavigationTarget::from_module_to_decl(db, module);
    vec![nav]
}

/// Returns `Vec` for the same reason as `parent_module`
pub(crate) fn crate_for(db: &RootDatabase, file_id: FileId) -> Vec<CrateId> {
    let source_file = db.parse(file_id).tree();
    let src = hir::ModuleSource::SourceFile(source_file);
    let module =
        match hir::Module::from_definition(db, hir::InFile { file_id: file_id.into(), value: src })
        {
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

    #[test]
    fn test_resolve_crate_root() {
        let mock = MockAnalysis::with_files(
            "
        //- /bar.rs
        mod foo;
        //- /foo.rs
        // empty <|>
    ",
        );
        let root_file = mock.id_of("/bar.rs");
        let mod_file = mock.id_of("/foo.rs");
        let mut host = mock.analysis_host();
        assert!(host.analysis().crate_for(mod_file).unwrap().is_empty());

        let mut crate_graph = CrateGraph::default();
        let crate_id = crate_graph.add_crate_root(
            root_file,
            Edition2018,
            CfgOptions::default(),
            Env::default(),
        );
        let mut change = AnalysisChange::new();
        change.set_crate_graph(crate_graph);
        host.apply_change(change);

        assert_eq!(host.analysis().crate_for(mod_file).unwrap(), vec![crate_id]);
    }
}
