//! A set of high-level utility fixture methods to use in tests.
use std::{mem, str::FromStr, sync::Arc};

use cfg::CfgOptions;
use rustc_hash::FxHashMap;
use test_utils::{
    extract_range_or_offset, Fixture, RangeOrOffset, CURSOR_MARKER, ESCAPED_CURSOR_MARKER,
};
use vfs::{file_set::FileSet, VfsPath};

use crate::{
    input::CrateName, Change, CrateGraph, CrateId, Edition, Env, FileId, FilePosition, FileRange,
    SourceDatabaseExt, SourceRoot, SourceRootId,
};

pub const WORKSPACE: SourceRootId = SourceRootId(0);

pub trait WithFixture: Default + SourceDatabaseExt + 'static {
    fn with_single_file(text: &str) -> (Self, FileId) {
        let fixture = ChangeFixture::parse(text);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        assert_eq!(fixture.files.len(), 1);
        (db, fixture.files[0])
    }

    fn with_files(ra_fixture: &str) -> Self {
        let fixture = ChangeFixture::parse(ra_fixture);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        assert!(fixture.file_position.is_none());
        db
    }

    fn with_position(ra_fixture: &str) -> (Self, FilePosition) {
        let (db, file_id, range_or_offset) = Self::with_range_or_offset(ra_fixture);
        let offset = match range_or_offset {
            RangeOrOffset::Range(_) => panic!("Expected a cursor position, got a range instead"),
            RangeOrOffset::Offset(it) => it,
        };
        (db, FilePosition { file_id, offset })
    }

    fn with_range(ra_fixture: &str) -> (Self, FileRange) {
        let (db, file_id, range_or_offset) = Self::with_range_or_offset(ra_fixture);
        let range = match range_or_offset {
            RangeOrOffset::Range(it) => it,
            RangeOrOffset::Offset(_) => panic!("Expected a cursor range, got a position instead"),
        };
        (db, FileRange { file_id, range })
    }

    fn with_range_or_offset(ra_fixture: &str) -> (Self, FileId, RangeOrOffset) {
        let fixture = ChangeFixture::parse(ra_fixture);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        let (file_id, range_or_offset) = fixture
            .file_position
            .expect("Could not find file position in fixture. Did you forget to add an `$0`?");
        (db, file_id, range_or_offset)
    }

    fn test_crate(&self) -> CrateId {
        let crate_graph = self.crate_graph();
        let mut it = crate_graph.iter();
        let res = it.next().unwrap();
        assert!(it.next().is_none());
        res
    }
}

impl<DB: SourceDatabaseExt + Default + 'static> WithFixture for DB {}

pub struct ChangeFixture {
    pub file_position: Option<(FileId, RangeOrOffset)>,
    pub files: Vec<FileId>,
    pub change: Change,
}

impl ChangeFixture {
    pub fn parse(ra_fixture: &str) -> ChangeFixture {
        let fixture = Fixture::parse(ra_fixture);
        let mut change = Change::new();

        let mut files = Vec::new();
        let mut crate_graph = CrateGraph::default();
        let mut crates = FxHashMap::default();
        let mut crate_deps = Vec::new();
        let mut default_crate_root: Option<FileId> = None;
        let mut default_cfg = CfgOptions::default();

        let mut file_set = FileSet::default();
        let source_root_prefix = "/".to_string();
        let mut file_id = FileId(0);
        let mut roots = Vec::new();

        let mut file_position = None;

        for entry in fixture {
            let text = if entry.text.contains(CURSOR_MARKER) {
                if entry.text.contains(ESCAPED_CURSOR_MARKER) {
                    entry.text.replace(ESCAPED_CURSOR_MARKER, CURSOR_MARKER)
                } else {
                    let (range_or_offset, text) = extract_range_or_offset(&entry.text);
                    assert!(file_position.is_none());
                    file_position = Some((file_id, range_or_offset));
                    text.to_string()
                }
            } else {
                entry.text.clone()
            };

            let meta = FileMeta::from(entry);
            assert!(meta.path.starts_with(&source_root_prefix));

            if meta.introduce_new_source_root {
                roots.push(SourceRoot::new_local(mem::take(&mut file_set)));
            }

            if let Some(krate) = meta.krate {
                let crate_name = CrateName::normalize_dashes(&krate);
                let crate_id = crate_graph.add_crate_root(
                    file_id,
                    meta.edition,
                    Some(crate_name.clone().into()),
                    meta.cfg,
                    meta.env,
                    Default::default(),
                );
                let prev = crates.insert(crate_name.clone(), crate_id);
                assert!(prev.is_none());
                for dep in meta.deps {
                    let dep = CrateName::normalize_dashes(&dep);
                    crate_deps.push((crate_name.clone(), dep))
                }
            } else if meta.path == "/main.rs" || meta.path == "/lib.rs" {
                assert!(default_crate_root.is_none());
                default_crate_root = Some(file_id);
                default_cfg = meta.cfg;
            }

            change.change_file(file_id, Some(Arc::new(text)));
            let path = VfsPath::new_virtual_path(meta.path);
            file_set.insert(file_id, path);
            files.push(file_id);
            file_id.0 += 1;
        }

        if crates.is_empty() {
            let crate_root = default_crate_root.unwrap();
            crate_graph.add_crate_root(
                crate_root,
                Edition::Edition2018,
                Some(CrateName::new("test").unwrap().into()),
                default_cfg,
                Env::default(),
                Default::default(),
            );
        } else {
            for (from, to) in crate_deps {
                let from_id = crates[&from];
                let to_id = crates[&to];
                crate_graph.add_dep(from_id, CrateName::new(&to).unwrap(), to_id).unwrap();
            }
        }

        roots.push(SourceRoot::new_local(mem::take(&mut file_set)));
        change.set_roots(roots);
        change.set_crate_graph(crate_graph);

        ChangeFixture { file_position, files, change }
    }
}

struct FileMeta {
    path: String,
    krate: Option<String>,
    deps: Vec<String>,
    cfg: CfgOptions,
    edition: Edition,
    env: Env,
    introduce_new_source_root: bool,
}

impl From<Fixture> for FileMeta {
    fn from(f: Fixture) -> FileMeta {
        let mut cfg = CfgOptions::default();
        f.cfg_atoms.iter().for_each(|it| cfg.insert_atom(it.into()));
        f.cfg_key_values.iter().for_each(|(k, v)| cfg.insert_key_value(k.into(), v.into()));

        FileMeta {
            path: f.path,
            krate: f.krate,
            deps: f.deps,
            cfg,
            edition: f
                .edition
                .as_ref()
                .map_or(Edition::Edition2018, |v| Edition::from_str(&v).unwrap()),
            env: f.env.into_iter().collect(),
            introduce_new_source_root: f.introduce_new_source_root,
        }
    }
}
