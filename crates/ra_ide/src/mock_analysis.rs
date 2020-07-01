//! FIXME: write short doc here
use std::sync::Arc;

use ra_cfg::CfgOptions;
use ra_db::{CrateName, Env, FileSet, SourceRoot, VfsPath};
use test_utils::{
    extract_annotations, extract_range_or_offset, Fixture, RangeOrOffset, CURSOR_MARKER,
};

use crate::{
    Analysis, AnalysisChange, AnalysisHost, CrateGraph, Edition, FileId, FilePosition, FileRange,
};

/// Mock analysis is used in test to bootstrap an AnalysisHost/Analysis
/// from a set of in-memory files.
#[derive(Debug, Default)]
pub struct MockAnalysis {
    files: Vec<Fixture>,
}

impl MockAnalysis {
    /// Creates `MockAnalysis` using a fixture data in the following format:
    ///
    /// ```not_rust
    /// //- /main.rs
    /// mod foo;
    /// fn main() {}
    ///
    /// //- /foo.rs
    /// struct Baz;
    /// ```
    pub fn with_files(ra_fixture: &str) -> MockAnalysis {
        let (res, pos) = MockAnalysis::with_fixture(ra_fixture);
        assert!(pos.is_none());
        res
    }

    /// Same as `with_files`, but requires that a single file contains a `<|>` marker,
    /// whose position is also returned.
    pub fn with_files_and_position(fixture: &str) -> (MockAnalysis, FilePosition) {
        let (res, position) = MockAnalysis::with_fixture(fixture);
        let (file_id, range_or_offset) = position.expect("expected a marker (<|>)");
        let offset = match range_or_offset {
            RangeOrOffset::Range(_) => panic!(),
            RangeOrOffset::Offset(it) => it,
        };
        (res, FilePosition { file_id, offset })
    }

    fn with_fixture(fixture: &str) -> (MockAnalysis, Option<(FileId, RangeOrOffset)>) {
        let mut position = None;
        let mut res = MockAnalysis::default();
        for mut entry in Fixture::parse(fixture) {
            if entry.text.contains(CURSOR_MARKER) {
                assert!(position.is_none(), "only one marker (<|>) per fixture is allowed");
                let (range_or_offset, text) = extract_range_or_offset(&entry.text);
                entry.text = text;
                let file_id = res.add_file_fixture(entry);
                position = Some((file_id, range_or_offset));
            } else {
                res.add_file_fixture(entry);
            }
        }
        (res, position)
    }

    fn add_file_fixture(&mut self, fixture: Fixture) -> FileId {
        let file_id = FileId((self.files.len() + 1) as u32);
        self.files.push(fixture);
        file_id
    }

    pub fn id_of(&self, path: &str) -> FileId {
        let (idx, _) = self
            .files
            .iter()
            .enumerate()
            .find(|(_, data)| path == data.path)
            .expect("no file in this mock");
        FileId(idx as u32 + 1)
    }
    pub fn annotations(&self) -> Vec<(FileRange, String)> {
        self.files
            .iter()
            .enumerate()
            .flat_map(|(idx, fixture)| {
                let file_id = FileId(idx as u32 + 1);
                let annotations = extract_annotations(&fixture.text);
                annotations
                    .into_iter()
                    .map(move |(range, data)| (FileRange { file_id, range }, data))
            })
            .collect()
    }
    pub fn annotation(&self) -> (FileRange, String) {
        let mut all = self.annotations();
        assert_eq!(all.len(), 1);
        all.pop().unwrap()
    }
    pub fn analysis_host(self) -> AnalysisHost {
        let mut host = AnalysisHost::default();
        let mut change = AnalysisChange::new();
        let mut file_set = FileSet::default();
        let mut crate_graph = CrateGraph::default();
        let mut root_crate = None;
        for (i, data) in self.files.into_iter().enumerate() {
            let path = data.path;
            assert!(path.starts_with('/'));

            let mut cfg = CfgOptions::default();
            data.cfg_atoms.iter().for_each(|it| cfg.insert_atom(it.into()));
            data.cfg_key_values.iter().for_each(|(k, v)| cfg.insert_key_value(k.into(), v.into()));
            let edition: Edition =
                data.edition.and_then(|it| it.parse().ok()).unwrap_or(Edition::Edition2018);

            let file_id = FileId(i as u32 + 1);
            let env = Env::from(data.env.iter());
            if path == "/lib.rs" || path == "/main.rs" {
                root_crate = Some(crate_graph.add_crate_root(
                    file_id,
                    edition,
                    None,
                    cfg,
                    env,
                    Default::default(),
                ));
            } else if path.ends_with("/lib.rs") {
                let base = &path[..path.len() - "/lib.rs".len()];
                let crate_name = &base[base.rfind('/').unwrap() + '/'.len_utf8()..];
                let other_crate = crate_graph.add_crate_root(
                    file_id,
                    edition,
                    Some(crate_name.to_string()),
                    cfg,
                    env,
                    Default::default(),
                );
                if let Some(root_crate) = root_crate {
                    crate_graph
                        .add_dep(root_crate, CrateName::new(crate_name).unwrap(), other_crate)
                        .unwrap();
                }
            }
            let path = VfsPath::new_virtual_path(path.to_string());
            file_set.insert(file_id, path);
            change.change_file(file_id, Some(Arc::new(data.text).to_owned()));
        }
        change.set_crate_graph(crate_graph);
        change.set_roots(vec![SourceRoot::new_local(file_set)]);
        host.apply_change(change);
        host
    }
    pub fn analysis(self) -> Analysis {
        self.analysis_host().analysis()
    }
}

/// Creates analysis from a multi-file fixture, returns positions marked with <|>.
pub fn analysis_and_position(ra_fixture: &str) -> (Analysis, FilePosition) {
    let (mock, position) = MockAnalysis::with_files_and_position(ra_fixture);
    (mock.analysis(), position)
}

/// Creates analysis for a single file.
pub fn single_file(ra_fixture: &str) -> (Analysis, FileId) {
    let mock = MockAnalysis::with_files(ra_fixture);
    let file_id = mock.id_of("/main.rs");
    (mock.analysis(), file_id)
}

/// Creates analysis for a single file, returns range marked with a pair of <|>.
pub fn analysis_and_range(ra_fixture: &str) -> (Analysis, FileRange) {
    let (res, position) = MockAnalysis::with_fixture(ra_fixture);
    let (file_id, range_or_offset) = position.expect("expected a marker (<|>)");
    let range = match range_or_offset {
        RangeOrOffset::Range(it) => it,
        RangeOrOffset::Offset(_) => panic!(),
    };
    (res.analysis(), FileRange { file_id, range })
}
