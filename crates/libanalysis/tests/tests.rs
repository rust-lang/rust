extern crate libanalysis;
extern crate relative_path;
extern crate test_utils;

use std::path::{Path};

use relative_path::RelativePath;
use libanalysis::{WorldState, FileId, FileResolver};
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
        let path = rel.to_path(self.path(id));
        let path = path.to_str().unwrap();
        let path = RelativePath::new(&path[1..]).normalize();
        let &(id, _) = self.0.iter()
            .find(|it| path == RelativePath::new(&it.1[1..]).normalize())?;
        Some(FileId(id))
    }
}


#[test]
fn test_resolve_module() {
    let mut world = WorldState::new();
    world.change_file(FileId(1), Some("mod foo;".to_string()));
    world.change_file(FileId(2), Some("".to_string()));

    let snap = world.snapshot(FileMap(&[
        (1, "/lib.rs"),
        (2, "/foo.rs"),
    ]));
    let symbols = snap.approximately_resolve_symbol(FileId(1), 4.into())
        .unwrap();
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );

    let snap = world.snapshot(FileMap(&[
        (1, "/lib.rs"),
        (2, "/foo/mod.rs")
    ]));
    let symbols = snap.approximately_resolve_symbol(FileId(1), 4.into())
        .unwrap();
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );
}

#[test]
fn test_unresolved_module_diagnostic() {
    let mut world = WorldState::new();
    world.change_file(FileId(1), Some("mod foo;".to_string()));

    let snap = world.snapshot(FileMap(&[(1, "/lib.rs")]));
    let diagnostics = snap.diagnostics(FileId(1)).unwrap();
    assert_eq_dbg(
        r#"[(Diagnostic { range: [4; 7), msg: "unresolved module" },
             Some(QuickFix { fs_ops: [CreateFile { anchor: FileId(1), path: "../foo.rs" }] }))]"#,
        &diagnostics,
    );
}

#[test]
fn test_unresolved_module_diagnostic_no_diag_for_inline_mode() {
    let mut world = WorldState::new();
    world.change_file(FileId(1), Some("mod foo {}".to_string()));

    let snap = world.snapshot(FileMap(&[(1, "/lib.rs")]));
    let diagnostics = snap.diagnostics(FileId(1)).unwrap();
    assert_eq_dbg(
        r#"[]"#,
        &diagnostics,
    );
}

#[test]
fn test_resolve_parent_module() {
    let mut world = WorldState::new();
    world.change_file(FileId(1), Some("mod foo;".to_string()));
    world.change_file(FileId(2), Some("".to_string()));

    let snap = world.snapshot(FileMap(&[
        (1, "/lib.rs"),
        (2, "/foo.rs"),
    ]));
    let symbols = snap.parent_module(FileId(2));
    assert_eq_dbg(
        r#"[(FileId(1), FileSymbol { name: "foo", node_range: [0; 8), kind: MODULE })]"#,
        &symbols,
    );
}
