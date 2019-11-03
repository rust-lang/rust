//! FIXME: write short doc here

use std::sync::Arc;

use ra_cfg::CfgOptions;
use rustc_hash::FxHashMap;
use test_utils::{extract_offset, parse_fixture, CURSOR_MARKER};

use crate::{
    CrateGraph, Edition, FileId, FilePosition, RelativePathBuf, SourceDatabaseExt, SourceRoot,
    SourceRootId,
};

pub const WORKSPACE: SourceRootId = SourceRootId(0);

pub trait WithFixture: Default + SourceDatabaseExt + 'static {
    fn with_single_file(text: &str) -> (Self, FileId) {
        let mut db = Self::default();
        let file_id = with_single_file(&mut db, text);
        (db, file_id)
    }

    fn with_files(fixture: &str) -> Self {
        let mut db = Self::default();
        let pos = with_files(&mut db, fixture);
        assert!(pos.is_none());
        db
    }

    fn with_position(fixture: &str) -> (Self, FilePosition) {
        let mut db = Self::default();
        let pos = with_files(&mut db, fixture);
        (db, pos.unwrap())
    }
}

impl<DB: SourceDatabaseExt + Default + 'static> WithFixture for DB {}

fn with_single_file(db: &mut dyn SourceDatabaseExt, text: &str) -> FileId {
    let file_id = FileId(0);
    let rel_path: RelativePathBuf = "/main.rs".into();

    let mut source_root = SourceRoot::default();
    source_root.insert_file(rel_path.clone(), file_id);

    let mut crate_graph = CrateGraph::default();
    crate_graph.add_crate_root(file_id, Edition::Edition2018, CfgOptions::default());

    db.set_file_text(file_id, Arc::new(text.to_string()));
    db.set_file_relative_path(file_id, rel_path);
    db.set_file_source_root(file_id, WORKSPACE);
    db.set_source_root(WORKSPACE, Arc::new(source_root));
    db.set_crate_graph(Arc::new(crate_graph));

    file_id
}

fn with_files(db: &mut dyn SourceDatabaseExt, fixture: &str) -> Option<FilePosition> {
    let fixture = parse_fixture(fixture);

    let mut crate_graph = CrateGraph::default();
    let mut crates = FxHashMap::default();
    let mut crate_deps = Vec::new();
    let mut default_crate_root: Option<FileId> = None;

    let mut source_root = SourceRoot::default();
    let mut source_root_id = WORKSPACE;
    let mut source_root_prefix: RelativePathBuf = "/".into();
    let mut file_id = FileId(0);

    let mut file_position = None;

    for entry in fixture.iter() {
        let meta = match parse_meta(&entry.meta) {
            ParsedMeta::Root { path } => {
                let source_root = std::mem::replace(&mut source_root, SourceRoot::default());
                db.set_source_root(source_root_id, Arc::new(source_root));
                source_root_id.0 += 1;
                source_root_prefix = path;
                continue;
            }
            ParsedMeta::File(it) => it,
        };
        assert!(meta.path.starts_with(&source_root_prefix));

        if let Some(krate) = meta.krate {
            let crate_id = crate_graph.add_crate_root(file_id, meta.edition, meta.cfg);
            let prev = crates.insert(krate.clone(), crate_id);
            assert!(prev.is_none());
            for dep in meta.deps {
                crate_deps.push((krate.clone(), dep))
            }
        } else if meta.path == "/main.rs" || meta.path == "/lib.rs" {
            assert!(default_crate_root.is_none());
            default_crate_root = Some(file_id);
        }

        let text = if entry.text.contains(CURSOR_MARKER) {
            let (offset, text) = extract_offset(&entry.text);
            assert!(file_position.is_none());
            file_position = Some(FilePosition { file_id, offset });
            text.to_string()
        } else {
            entry.text.to_string()
        };

        db.set_file_text(file_id, Arc::new(text));
        db.set_file_relative_path(file_id, meta.path.clone());
        db.set_file_source_root(file_id, source_root_id);
        source_root.insert_file(meta.path, file_id);

        file_id.0 += 1;
    }

    if crates.is_empty() {
        let crate_root = default_crate_root.unwrap();
        crate_graph.add_crate_root(crate_root, Edition::Edition2018, CfgOptions::default());
    } else {
        for (from, to) in crate_deps {
            let from_id = crates[&from];
            let to_id = crates[&to];
            crate_graph.add_dep(from_id, to.into(), to_id).unwrap();
        }
    }

    db.set_source_root(source_root_id, Arc::new(source_root));
    db.set_crate_graph(Arc::new(crate_graph));

    file_position
}

enum ParsedMeta {
    Root { path: RelativePathBuf },
    File(FileMeta),
}

struct FileMeta {
    path: RelativePathBuf,
    krate: Option<String>,
    deps: Vec<String>,
    cfg: CfgOptions,
    edition: Edition,
}

//- /lib.rs crate:foo deps:bar,baz
fn parse_meta(meta: &str) -> ParsedMeta {
    let components = meta.split_ascii_whitespace().collect::<Vec<_>>();

    if components[0] == "root" {
        let path: RelativePathBuf = components[1].into();
        assert!(path.starts_with("/") && path.ends_with("/"));
        return ParsedMeta::Root { path };
    }

    let path: RelativePathBuf = components[0].into();
    assert!(path.starts_with("/"));

    let mut krate = None;
    let mut deps = Vec::new();
    let mut edition = Edition::Edition2018;
    let mut cfg = CfgOptions::default();
    for component in components[1..].iter() {
        let (key, value) = split1(component, ':').unwrap();
        match key {
            "crate" => krate = Some(value.to_string()),
            "deps" => deps = value.split(',').map(|it| it.to_string()).collect(),
            "edition" => edition = Edition::from_string(&value),
            "cfg" => {
                for key in value.split(',') {
                    match split1(key, '=') {
                        None => cfg.insert_atom(key.into()),
                        Some((k, v)) => cfg.insert_key_value(k.into(), v.into()),
                    }
                }
            }
            _ => panic!("bad component: {:?}", component),
        }
    }

    ParsedMeta::File(FileMeta { path, krate, deps, edition, cfg })
}

fn split1(haystack: &str, delim: char) -> Option<(&str, &str)> {
    let idx = haystack.find(delim)?;
    Some((&haystack[..idx], &haystack[idx + delim.len_utf8()..]))
}
