extern crate relative_path;
extern crate ra_analysis;
extern crate rustc_hash;
extern crate ra_editor;
extern crate ra_syntax;
extern crate test_utils;

use std::{
    sync::Arc,
};

use rustc_hash::FxHashMap;
use relative_path::{RelativePath, RelativePathBuf};
use ra_analysis::{Analysis, AnalysisHost, FileId, FileResolver, JobHandle, CrateGraph, CrateId, FnDescriptor};
use test_utils::{assert_eq_dbg, extract_offset};

#[derive(Debug)]
struct FileMap(Vec<(FileId, RelativePathBuf)>);

impl FileMap {
    fn iter<'a>(&'a self) -> impl Iterator<Item=(FileId, &'a RelativePath)> + 'a {
        self.0.iter().map(|(id, path)| (*id, path.as_relative_path()))
    }

    fn path(&self, id: FileId) -> &RelativePath {
        self.iter()
            .find(|&(it, _)| it == id)
            .unwrap()
            .1
    }
}

impl FileResolver for FileMap {
    fn file_stem(&self, id: FileId) -> String {
        self.path(id).file_stem().unwrap().to_string()
    }
    fn resolve(&self, id: FileId, rel: &RelativePath) -> Option<FileId> {
        let path = self.path(id).join(rel).normalize();
        let id = self.iter().find(|&(_, p)| path == p)?.0;
        Some(id)
    }
}

fn analysis_host(files: &[(&str, &str)]) -> AnalysisHost {
    let mut host = AnalysisHost::new();
    let mut file_map = Vec::new();
    for (id, &(path, contents)) in files.iter().enumerate() {
        let file_id = FileId((id + 1) as u32);
        assert!(path.starts_with('/'));
        let path = RelativePathBuf::from_path(&path[1..]).unwrap();
        host.change_file(file_id, Some(contents.to_string()));
        file_map.push((file_id, path));
    }
    host.set_file_resolver(Arc::new(FileMap(file_map)));
    host
}

fn analysis(files: &[(&str, &str)]) -> Analysis {
    analysis_host(files).analysis()
}

fn get_signature(text: &str) -> (FnDescriptor, Option<usize>) {
    let (offset, code) = extract_offset(text);
    let code = code.as_str();

    let (_handle, token) = JobHandle::new();
    let snap = analysis(&[("/lib.rs", code)]);

    snap.resolve_callable(FileId(1), offset, &token).unwrap()
}

#[test]
fn test_resolve_module() {
    let snap = analysis(&[
        ("/lib.rs", "mod foo;"),
        ("/foo.rs", "")
    ]);
    let (_handle, token) = JobHandle::new();
    let symbols = snap.approximately_resolve_symbol(FileId(1), 4.into(), &token);
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );

    let snap = analysis(&[
        ("/lib.rs", "mod foo;"),
        ("/foo/mod.rs", "")
    ]);
    let symbols = snap.approximately_resolve_symbol(FileId(1), 4.into(), &token);
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );
}

#[test]
fn test_unresolved_module_diagnostic() {
    let snap = analysis(&[("/lib.rs", "mod foo;")]);
    let diagnostics = snap.diagnostics(FileId(1));
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
    let snap = analysis(&[("/lib.rs", "mod foo {}")]);
    let diagnostics = snap.diagnostics(FileId(1));
    assert_eq_dbg(
        r#"[]"#,
        &diagnostics,
    );
}

#[test]
fn test_resolve_parent_module() {
    let snap = analysis(&[
        ("/lib.rs", "mod foo;"),
        ("/foo.rs", ""),
    ]);
    let symbols = snap.parent_module(FileId(2));
    assert_eq_dbg(
        r#"[(FileId(1), FileSymbol { name: "foo", node_range: [0; 8), kind: MODULE })]"#,
        &symbols,
    );
}

#[test]
fn test_resolve_crate_root() {
    let mut host = analysis_host(&[
        ("/lib.rs", "mod foo;"),
        ("/foo.rs", ""),
    ]);
    let snap = host.analysis();
    assert!(snap.crate_for(FileId(2)).is_empty());

    let crate_graph = CrateGraph {
        crate_roots: {
            let mut m = FxHashMap::default();
            m.insert(CrateId(1), FileId(1));
            m
        },
    };
    host.set_crate_graph(crate_graph);
    let snap = host.analysis();

    assert_eq!(
        snap.crate_for(FileId(2)),
        vec![CrateId(1)],
    );
}

#[test]
fn test_fn_signature_two_args_first() {
    let (desc, param) = get_signature(
r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(<|>3, ); }"#);

    assert_eq!(desc.name, "foo".to_string());
    assert_eq!(desc.params, vec!("x".to_string(),"y".to_string()));
    assert_eq!(desc.ret_type, Some("-> u32".into()));
    assert_eq!(param, Some(0));
}

#[test]
fn test_fn_signature_two_args_second() {
    let (desc, param) = get_signature(
        r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3, <|>); }"#);

    assert_eq!(desc.name, "foo".to_string());
    assert_eq!(desc.params, vec!("x".to_string(),"y".to_string()));
    assert_eq!(desc.ret_type, Some("-> u32".into()));
    assert_eq!(param, Some(1));
}

#[test]
fn test_fn_signature_for_impl() {
    let (desc, param) = get_signature(
r#"struct F; impl F { pub fn new() { F{}} }
fn bar() {let _ : F = F::new(<|>);}"#);

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
}"#);

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
}"#);

    assert_eq!(desc.name, "do_it".to_string());
    assert_eq!(desc.params, vec!["&self".to_string(), "x".to_string()]);
    assert_eq!(desc.ret_type, None);
    assert_eq!(param, Some(1));
}