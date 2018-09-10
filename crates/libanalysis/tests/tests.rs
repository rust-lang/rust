extern crate libanalysis;
extern crate relative_path;
extern crate test_utils;

use std::{
    sync::Arc,
    collections::HashMap,
};

use relative_path::{RelativePath, RelativePathBuf};
use libanalysis::{Analysis, AnalysisHost, FileId, FileResolver, JobHandle, CrateGraph, CrateId};
use test_utils::assert_eq_dbg;

#[derive(Debug)]
struct FileMap(Vec<(FileId, RelativePathBuf)>);

fn analysis_host(files: &'static [(&'static str, &'static str)]) -> AnalysisHost {
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

fn analysis(files: &'static [(&'static str, &'static str)]) -> Analysis {
    analysis_host(files).analysis()
}

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
            let mut m = HashMap::new();
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
