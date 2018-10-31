extern crate ra_analysis;
extern crate ra_editor;
extern crate ra_syntax;
extern crate relative_path;
extern crate rustc_hash;
extern crate test_utils;

use ra_syntax::{TextRange};
use test_utils::{assert_eq_dbg};

use ra_analysis::{
    AnalysisChange, CrateGraph, FileId, FnDescriptor,
    mock_analysis::{MockAnalysis, single_file, single_file_with_position, analysis_and_position},
};

fn get_signature(text: &str) -> (FnDescriptor, Option<usize>) {
    let (analysis, position) = single_file_with_position(text);
    analysis.resolve_callable(position.file_id, position.offset).unwrap().unwrap()
}

#[test]
fn test_resolve_module() {
    let (analysis, pos) = analysis_and_position("
        //- /lib.rs
        mod <|>foo;
        //- /foo.rs
        // empty
    ");

    let symbols = analysis.approximately_resolve_symbol(pos.file_id, pos.offset).unwrap();
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );

    let (analysis, pos) = analysis_and_position("
        //- /lib.rs
        mod <|>foo;
        //- /foo/mod.rs
        // empty
    ");

    let symbols = analysis.approximately_resolve_symbol(pos.file_id, pos.offset).unwrap();
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );
}

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
                file_system_edits: [CreateFile { anchor: FileId(1), path: "../foo.rs" }],
                cursor_position: None }) }]"#,
        &diagnostics,
    );
}

#[test]
fn test_unresolved_module_diagnostic_no_diag_for_inline_mode() {
    let (analysis, file_id) = single_file("mod foo {}");
    let diagnostics = analysis.diagnostics(file_id).unwrap();
    assert_eq_dbg(r#"[]"#, &diagnostics);
}

#[test]
fn test_resolve_parent_module() {
    let (analysis, pos) = analysis_and_position("
        //- /lib.rs
        mod foo;
        //- /foo.rs
        <|>// empty
    ");
    let symbols = analysis.parent_module(pos.file_id).unwrap();
    assert_eq_dbg(
        r#"[(FileId(1), FileSymbol { name: "foo", node_range: [0; 8), kind: MODULE })]"#,
        &symbols,
    );
}

#[test]
fn test_resolve_crate_root() {
    let mock = MockAnalysis::with_files("
        //- /lib.rs
        mod foo;
        //- /foo.rs
        // emtpy <|>
    ");
    let root_file = mock.id_of("/lib.rs");
    let mod_file = mock.id_of("/foo.rs");
    let mut host = mock.analysis_host();
    assert!(host.analysis().crate_for(mod_file).unwrap().is_empty());

    let mut crate_graph = CrateGraph::new();
    let crate_id = crate_graph.add_crate_root(root_file);
    let mut change = AnalysisChange::new();
    change.set_crate_graph(crate_graph);
    host.apply_change(change);

    assert_eq!(host.analysis().crate_for(mod_file).unwrap(), vec![crate_id]);
}

#[test]
fn test_fn_signature_two_args_first() {
    let (desc, param) = get_signature(
        r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(<|>3, ); }"#,
    );

    assert_eq!(desc.name, "foo".to_string());
    assert_eq!(desc.params, vec!("x".to_string(), "y".to_string()));
    assert_eq!(desc.ret_type, Some("-> u32".into()));
    assert_eq!(param, Some(0));
}

#[test]
fn test_fn_signature_two_args_second() {
    let (desc, param) = get_signature(
        r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3, <|>); }"#,
    );

    assert_eq!(desc.name, "foo".to_string());
    assert_eq!(desc.params, vec!("x".to_string(), "y".to_string()));
    assert_eq!(desc.ret_type, Some("-> u32".into()));
    assert_eq!(param, Some(1));
}

#[test]
fn test_fn_signature_for_impl() {
    let (desc, param) = get_signature(
        r#"struct F; impl F { pub fn new() { F{}} }
fn bar() {let _ : F = F::new(<|>);}"#,
    );

    assert_eq!(desc.name, "new".to_string());
    assert_eq!(desc.params, Vec::<String>::new());
    assert_eq!(desc.ret_type, None);
    assert_eq!(param, None);
}

#[test]
fn test_fn_signature_for_method_self() {
    let (desc, param) = get_signature(
        r#"struct F;
impl F {
    pub fn new() -> F{
        F{}
    }

    pub fn do_it(&self) {}
}

fn bar() {
    let f : F = F::new();
    f.do_it(<|>);
}"#,
    );

    assert_eq!(desc.name, "do_it".to_string());
    assert_eq!(desc.params, vec!["&self".to_string()]);
    assert_eq!(desc.ret_type, None);
    assert_eq!(param, None);
}

#[test]
fn test_fn_signature_for_method_with_arg() {
    let (desc, param) = get_signature(
        r#"struct F;
impl F {
    pub fn new() -> F{
        F{}
    }

    pub fn do_it(&self, x: i32) {}
}

fn bar() {
    let f : F = F::new();
    f.do_it(<|>);
}"#,
    );

    assert_eq!(desc.name, "do_it".to_string());
    assert_eq!(desc.params, vec!["&self".to_string(), "x".to_string()]);
    assert_eq!(desc.ret_type, None);
    assert_eq!(param, Some(1));
}

fn get_all_refs(text: &str) -> Vec<(FileId, TextRange)> {
    let (analysis, position) = single_file_with_position(text);
    analysis.find_all_refs(position.file_id, position.offset).unwrap()
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
fn test_complete_crate_path() {
    let (analysis, position) = analysis_and_position("
        //- /lib.rs
        mod foo;
        struct Spam;
        //- /foo.rs
        use crate::Sp<|>
    ");
    let completions = analysis.completions(position.file_id, position.offset).unwrap().unwrap();
    assert_eq_dbg(
        r#"[CompletionItem { label: "foo", lookup: None, snippet: None },
            CompletionItem { label: "Spam", lookup: None, snippet: None }]"#,
        &completions,
    );
}
