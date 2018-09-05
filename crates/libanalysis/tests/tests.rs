extern crate libanalysis;
extern crate relative_path;
extern crate test_utils;

use std::{
    collections::HashMap,
    path::{Path},
};

use relative_path::RelativePath;
use libanalysis::{AnalysisHost, FileId, FileResolver, JobHandle, CrateGraph, CrateId};
use test_utils::assert_eq_dbg;

struct FileMap(&'static [(u32, &'static str)]);

impl FileMap {
    fn path(&self, id: FileId) -> &'static Path {
        let s = self.0.iter()
            .find(|it| it.0 == id.0)
            .unwrap()
            .1;
        Path::new(s)
    }
}

impl FileResolver for FileMap {
    fn file_stem(&self, id: FileId) -> String {
        self.path(id).file_stem().unwrap().to_str().unwrap().to_string()
    }
    fn resolve(&self, id: FileId, rel: &RelativePath) -> Option<FileId> {
        let path = {
            if rel.starts_with("..") {
                rel.strip_prefix("..").unwrap()
                    .to_path(&self.path(id).parent().unwrap())
            } else {
                rel.to_path(self.path(id))
            }
        };
        let path = &path.to_str().unwrap()[1..];
        let path = RelativePath::new(&path[0..]).normalize();
        let &(id, _) = self.0.iter()
            .find(|it| path == RelativePath::new(&it.1[0..]).normalize())?;
        Some(FileId(id))
    }
}


#[test]
fn test_resolve_module() {
    let mut world = AnalysisHost::new();
    world.change_file(FileId(1), Some("mod foo;".to_string()));
    world.change_file(FileId(2), Some("".to_string()));

    let snap = world.analysis(FileMap(&[
        (1, "/lib.rs"),
        (2, "/foo.rs"),
    ]));
    let (_handle, token) = JobHandle::new();
    let symbols = snap.approximately_resolve_symbol(FileId(1), 4.into(), &token);
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );

    let snap = world.analysis(FileMap(&[
        (1, "/lib.rs"),
        (2, "/foo/mod.rs")
    ]));
    let symbols = snap.approximately_resolve_symbol(FileId(1), 4.into(), &token);
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );
}

#[test]
fn test_unresolved_module_diagnostic() {
    let mut world = AnalysisHost::new();
    world.change_file(FileId(1), Some("mod foo;".to_string()));

    let snap = world.analysis(FileMap(&[(1, "/lib.rs")]));
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
    let mut world = AnalysisHost::new();
    world.change_file(FileId(1), Some("mod foo {}".to_string()));

    let snap = world.analysis(FileMap(&[(1, "/lib.rs")]));
    let diagnostics = snap.diagnostics(FileId(1));
    assert_eq_dbg(
        r#"[]"#,
        &diagnostics,
    );
}

#[test]
fn test_resolve_parent_module() {
    let mut world = AnalysisHost::new();
    world.change_file(FileId(1), Some("mod foo;".to_string()));
    world.change_file(FileId(2), Some("".to_string()));

    let snap = world.analysis(FileMap(&[
        (1, "/lib.rs"),
        (2, "/foo.rs"),
    ]));
    let symbols = snap.parent_module(FileId(2));
    assert_eq_dbg(
        r#"[(FileId(1), FileSymbol { name: "foo", node_range: [0; 8), kind: MODULE })]"#,
        &symbols,
    );
}

#[test]
fn test_resolve_crate_root() {
    let mut world = AnalysisHost::new();
    world.change_file(FileId(1), Some("mod foo;".to_string()));
    world.change_file(FileId(2), Some("".to_string()));

    let snap = world.analysis(FileMap(&[
        (1, "/lib.rs"),
        (2, "/foo.rs"),
    ]));
    assert!(snap.crate_for(FileId(2)).is_empty());

    let crate_graph = CrateGraph {
        crate_roots: {
            let mut m = HashMap::new();
            m.insert(CrateId(1), FileId(1));
            m
        },
    };
    world.set_crate_graph(crate_graph);

    let snap = world.analysis(FileMap(&[
        (1, "/lib.rs"),
        (2, "/foo.rs"),
    ]));
    assert_eq!(
        snap.crate_for(FileId(2)),
        vec![CrateId(1)],
    );
}
