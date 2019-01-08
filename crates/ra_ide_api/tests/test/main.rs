mod runnables;

use ra_syntax::TextRange;
use test_utils::{assert_eq_dbg, assert_eq_text};

use ra_ide_api::{
    mock_analysis::{analysis_and_position, single_file, single_file_with_position, MockAnalysis},
    AnalysisChange, CrateGraph, FileId, Query
};

#[test]
fn test_unresolved_module_diagnostic() {
    let (analysis, file_id) = single_file("mod foo;");
    let diagnostics = analysis.diagnostics(file_id).unwrap();
    assert_eq_dbg(
        r#"[Diagnostic {
            message: "unresolved module",
            range: [4; 7),
            fix: Some(SourceChange {
                label: "create module",
                source_file_edits: [],
                file_system_edits: [CreateFile { source_root: SourceRootId(0), path: "foo.rs" }],
                cursor_position: None }),
                severity: Error }]"#,
        &diagnostics,
    );
}

// FIXME: move this test to hir
#[test]
fn test_unresolved_module_diagnostic_no_diag_for_inline_mode() {
    let (analysis, file_id) = single_file("mod foo {}");
    let diagnostics = analysis.diagnostics(file_id).unwrap();
    assert_eq_dbg(r#"[]"#, &diagnostics);
}

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
    let symbols = analysis.parent_module(pos).unwrap();
    assert_eq_dbg(
        r#"[NavigationTarget { file_id: FileId(1), name: "foo", kind: MODULE, range: [4; 7), ptr: None }]"#,
        &symbols,
    );
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
    let symbols = analysis.parent_module(pos).unwrap();
    assert_eq_dbg(
        r#"[NavigationTarget { file_id: FileId(1), name: "baz", kind: MODULE, range: [36; 39), ptr: None }]"#,
        &symbols,
    );
}

#[test]
fn test_resolve_crate_root() {
    let mock = MockAnalysis::with_files(
        "
        //- /bar.rs
        mod foo;
        //- /bar/foo.rs
        // emtpy <|>
    ",
    );
    let root_file = mock.id_of("/bar.rs");
    let mod_file = mock.id_of("/bar/foo.rs");
    let mut host = mock.analysis_host();
    assert!(host.analysis().crate_for(mod_file).unwrap().is_empty());

    let mut crate_graph = CrateGraph::default();
    let crate_id = crate_graph.add_crate_root(root_file);
    let mut change = AnalysisChange::new();
    change.set_crate_graph(crate_graph);
    host.apply_change(change);

    assert_eq!(host.analysis().crate_for(mod_file).unwrap(), vec![crate_id]);
}

fn get_all_refs(text: &str) -> Vec<(FileId, TextRange)> {
    let (analysis, position) = single_file_with_position(text);
    analysis.find_all_refs(position).unwrap()
}

#[test]
fn test_find_all_refs_for_local() {
    let code = r#"
    fn main() {
        let mut i = 1;
        let j = 1;
        i = i<|> + j;

        {
            i = 0;
        }

        i = 5;
    }"#;

    let refs = get_all_refs(code);
    assert_eq!(refs.len(), 5);
}

#[test]
fn test_find_all_refs_for_param_inside() {
    let code = r#"
    fn foo(i : u32) -> u32 {
        i<|>
    }"#;

    let refs = get_all_refs(code);
    assert_eq!(refs.len(), 2);
}

#[test]
fn test_find_all_refs_for_fn_param() {
    let code = r#"
    fn foo(i<|> : u32) -> u32 {
        i
    }"#;

    let refs = get_all_refs(code);
    assert_eq!(refs.len(), 2);
}
#[test]
fn test_rename_for_local() {
    test_rename(
        r#"
    fn main() {
        let mut i = 1;
        let j = 1;
        i = i<|> + j;

        {
            i = 0;
        }

        i = 5;
    }"#,
        "k",
        r#"
    fn main() {
        let mut k = 1;
        let j = 1;
        k = k + j;

        {
            k = 0;
        }

        k = 5;
    }"#,
    );
}

#[test]
fn test_rename_for_param_inside() {
    test_rename(
        r#"
    fn foo(i : u32) -> u32 {
        i<|>
    }"#,
        "j",
        r#"
    fn foo(j : u32) -> u32 {
        j
    }"#,
    );
}

#[test]
fn test_rename_refs_for_fn_param() {
    test_rename(
        r#"
    fn foo(i<|> : u32) -> u32 {
        i
    }"#,
        "new_name",
        r#"
    fn foo(new_name : u32) -> u32 {
        new_name
    }"#,
    );
}

#[test]
fn test_rename_for_mut_param() {
    test_rename(
        r#"
    fn foo(mut i<|> : u32) -> u32 {
        i
    }"#,
        "new_name",
        r#"
    fn foo(mut new_name : u32) -> u32 {
        new_name
    }"#,
    );
}

fn test_rename(text: &str, new_name: &str, expected: &str) {
    let (analysis, position) = single_file_with_position(text);
    let edits = analysis.rename(position, new_name).unwrap();
    let mut text_edit_bulder = ra_text_edit::TextEditBuilder::default();
    let mut file_id: Option<FileId> = None;
    for edit in edits {
        file_id = Some(edit.file_id);
        for atom in edit.edit.as_atoms() {
            text_edit_bulder.replace(atom.delete, atom.insert.clone());
        }
    }
    let result = text_edit_bulder
        .finish()
        .apply(&*analysis.file_text(file_id.unwrap()));
    assert_eq_text!(expected, &*result);
}

#[test]
fn world_symbols_include_stuff_from_macros() {
    let (analysis, _) = single_file(
        "
salsa::query_group! {
pub trait HirDatabase: SyntaxDatabase {}
}
    ",
    );

    let mut symbols = analysis.symbol_search(Query::new("Hir".into())).unwrap();
    let s = symbols.pop().unwrap();
    assert_eq!(s.name(), "HirDatabase");
    assert_eq!(s.range(), TextRange::from_to(33.into(), 44.into()));
}
