extern crate libanalysis;
extern crate assert_eq_text;

use std::path::PathBuf;

use libanalysis::{WorldState, FileId};
use assert_eq_text::assert_eq_dbg;


#[test]
fn test_resolve_module() {
    let mut world = WorldState::new();
    world.change_file(FileId(1), Some("mod foo;".to_string()));
    world.change_file(FileId(2), Some("".to_string()));

    let snap = world.snapshot(|id, path| {
        assert_eq!(id, FileId(1));
        if path == PathBuf::from("../foo/mod.rs") {
            return None;
        }
        assert_eq!(path, PathBuf::from("../foo.rs"));
        Some(FileId(2))
    });
    let symbols = snap.approximately_resolve_symbol(FileId(1), 4.into())
        .unwrap();
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );

    let snap = world.snapshot(|id, path| {
        assert_eq!(id, FileId(1));
        if path == PathBuf::from("../foo.rs") {
            return None;
        }
        assert_eq!(path, PathBuf::from("../foo/mod.rs"));
        Some(FileId(2))
    });
    let symbols = snap.approximately_resolve_symbol(FileId(1), 4.into())
        .unwrap();
    assert_eq_dbg(
        r#"[(FileId(2), FileSymbol { name: "foo", node_range: [0; 0), kind: MODULE })]"#,
        &symbols,
    );
}
