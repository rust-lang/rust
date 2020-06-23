//! Fixtures are strings containing rust source code with optional metadata.
//! A fixture without metadata is parsed into a single source file.
//! Use this to test functionality local to one file.
//!
//! Simple Example:
//! ```
//! r#"
//! fn main() {
//!     println!("Hello World")
//! }
//! "#
//! ```
//!
//! Metadata can be added to a fixture after a `//-` comment.
//! The basic form is specifying filenames,
//! which is also how to define multiple files in a single test fixture
//!
//! Example using two files in the same crate:
//! ```
//! "
//! //- /main.rs
//! mod foo;
//! fn main() {
//!     foo::bar();
//! }
//!
//! //- /foo.rs
//! pub fn bar() {}
//! "
//! ```
//!
//! Example using two crates with one file each, with one crate depending on the other:
//! ```
//! r#"
//! //- /main.rs crate:a deps:b
//! fn main() {
//!     b::foo();
//! }
//! //- /lib.rs crate:b
//! pub fn b() {
//!     println!("Hello World")
//! }
//! "#
//! ```
//!
//! Metadata allows specifying all settings and variables
//! that are available in a real rust project:
//! - crate names via `crate:cratename`
//! - dependencies via `deps:dep1,dep2`
//! - configuration settings via `cfg:dbg=false,opt_level=2`
//! - environment variables via `env:PATH=/bin,RUST_LOG=debug`
//!
//! Example using all available metadata:
//! ```
//! "
//! //- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b env:OUTDIR=path/to,OTHER=foo
//! fn insert_source_code_here() {}
//! "
//! ```
use std::{str::FromStr, sync::Arc};

use ra_cfg::CfgOptions;
use rustc_hash::FxHashMap;
use test_utils::{extract_range_or_offset, Fixture, RangeOrOffset, CURSOR_MARKER};
use vfs::{file_set::FileSet, VfsPath};

use crate::{
    input::CrateName, CrateGraph, CrateId, Edition, Env, FileId, FilePosition, SourceDatabaseExt,
    SourceRoot, SourceRootId,
};

pub const WORKSPACE: SourceRootId = SourceRootId(0);

pub trait WithFixture: Default + SourceDatabaseExt + 'static {
    fn with_single_file(text: &str) -> (Self, FileId) {
        let mut db = Self::default();
        let file_id = with_single_file(&mut db, text);
        (db, file_id)
    }

    fn with_files(ra_fixture: &str) -> Self {
        let mut db = Self::default();
        let (pos, _) = with_files(&mut db, ra_fixture);
        assert!(pos.is_none());
        db
    }

    fn with_position(ra_fixture: &str) -> (Self, FilePosition) {
        let (db, file_id, range_or_offset) = Self::with_range_or_offset(ra_fixture);
        let offset = match range_or_offset {
            RangeOrOffset::Range(_) => panic!(),
            RangeOrOffset::Offset(it) => it,
        };
        (db, FilePosition { file_id, offset })
    }

    fn with_range_or_offset(ra_fixture: &str) -> (Self, FileId, RangeOrOffset) {
        let mut db = Self::default();
        let (pos, _) = with_files(&mut db, ra_fixture);
        let (file_id, range_or_offset) = pos.unwrap();
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

fn with_single_file(db: &mut dyn SourceDatabaseExt, ra_fixture: &str) -> FileId {
    let file_id = FileId(0);
    let mut file_set = vfs::file_set::FileSet::default();
    file_set.insert(file_id, vfs::VfsPath::new_virtual_path("/main.rs".to_string()));

    let source_root = SourceRoot::new_local(file_set);

    let crate_graph = if let Some(meta) = ra_fixture.lines().find(|it| it.contains("//-")) {
        let entry = Fixture::parse_meta_line(meta.trim());
        let meta = match ParsedMeta::from(&entry) {
            ParsedMeta::File(it) => it,
        };

        let mut crate_graph = CrateGraph::default();
        crate_graph.add_crate_root(
            file_id,
            meta.edition,
            meta.krate.map(|name| {
                CrateName::new(&name).expect("Fixture crate name should not contain dashes")
            }),
            meta.cfg,
            meta.env,
            Default::default(),
        );
        crate_graph
    } else {
        let mut crate_graph = CrateGraph::default();
        crate_graph.add_crate_root(
            file_id,
            Edition::Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        crate_graph
    };

    db.set_file_text(file_id, Arc::new(ra_fixture.to_string()));
    db.set_file_source_root(file_id, WORKSPACE);
    db.set_source_root(WORKSPACE, Arc::new(source_root));
    db.set_crate_graph(Arc::new(crate_graph));

    file_id
}

fn with_files(
    db: &mut dyn SourceDatabaseExt,
    fixture: &str,
) -> (Option<(FileId, RangeOrOffset)>, Vec<FileId>) {
    let fixture = Fixture::parse(fixture);

    let mut files = Vec::new();
    let mut crate_graph = CrateGraph::default();
    let mut crates = FxHashMap::default();
    let mut crate_deps = Vec::new();
    let mut default_crate_root: Option<FileId> = None;

    let mut file_set = FileSet::default();
    let source_root_id = WORKSPACE;
    let source_root_prefix = "/".to_string();
    let mut file_id = FileId(0);

    let mut file_position = None;

    for entry in fixture.iter() {
        let meta = match ParsedMeta::from(entry) {
            ParsedMeta::File(it) => it,
        };
        assert!(meta.path.starts_with(&source_root_prefix));

        if let Some(krate) = meta.krate {
            let crate_id = crate_graph.add_crate_root(
                file_id,
                meta.edition,
                Some(CrateName::new(&krate).unwrap()),
                meta.cfg,
                meta.env,
                Default::default(),
            );
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
            let (range_or_offset, text) = extract_range_or_offset(&entry.text);
            assert!(file_position.is_none());
            file_position = Some((file_id, range_or_offset));
            text.to_string()
        } else {
            entry.text.to_string()
        };

        db.set_file_text(file_id, Arc::new(text));
        db.set_file_source_root(file_id, source_root_id);
        let path = VfsPath::new_virtual_path(meta.path);
        file_set.insert(file_id, path.into());
        files.push(file_id);
        file_id.0 += 1;
    }

    if crates.is_empty() {
        let crate_root = default_crate_root.unwrap();
        crate_graph.add_crate_root(
            crate_root,
            Edition::Edition2018,
            None,
            CfgOptions::default(),
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

    db.set_source_root(source_root_id, Arc::new(SourceRoot::new_local(file_set)));
    db.set_crate_graph(Arc::new(crate_graph));

    (file_position, files)
}

enum ParsedMeta {
    File(FileMeta),
}

struct FileMeta {
    path: String,
    krate: Option<String>,
    deps: Vec<String>,
    cfg: CfgOptions,
    edition: Edition,
    env: Env,
}

impl From<&Fixture> for ParsedMeta {
    fn from(f: &Fixture) -> Self {
        let mut cfg = CfgOptions::default();
        f.cfg_atoms.iter().for_each(|it| cfg.insert_atom(it.into()));
        f.cfg_key_values.iter().for_each(|(k, v)| cfg.insert_key_value(k.into(), v.into()));

        Self::File(FileMeta {
            path: f.path.to_owned(),
            krate: f.crate_name.to_owned(),
            deps: f.deps.to_owned(),
            cfg,
            edition: f
                .edition
                .as_ref()
                .map_or(Edition::Edition2018, |v| Edition::from_str(&v).unwrap()),
            env: Env::from(f.env.iter()),
        })
    }
}
