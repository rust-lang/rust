//! FIXME: write short doc here
use std::{str::FromStr, sync::Arc};

use ra_cfg::CfgOptions;
use ra_db::{CrateName, Env, FileSet, SourceRoot, VfsPath};
use test_utils::{extract_offset, extract_range, Fixture, CURSOR_MARKER};

use crate::{
    Analysis, AnalysisChange, AnalysisHost, CrateGraph, Edition, FileId, FilePosition, FileRange,
};

#[derive(Debug)]
enum MockFileData {
    Plain { path: String, content: String },
    Fixture(Fixture),
}

impl MockFileData {
    fn new(path: String, content: String) -> Self {
        // `Self::Plain` causes a false warning: 'variant is never constructed: `Plain` '
        // see https://github.com/rust-lang/rust/issues/69018
        MockFileData::Plain { path, content }
    }

    fn path(&self) -> &str {
        match self {
            MockFileData::Plain { path, .. } => path.as_str(),
            MockFileData::Fixture(f) => f.path.as_str(),
        }
    }

    fn content(&self) -> &str {
        match self {
            MockFileData::Plain { content, .. } => content,
            MockFileData::Fixture(f) => f.text.as_str(),
        }
    }

    fn cfg_options(&self) -> CfgOptions {
        match self {
            MockFileData::Fixture(f) => f.cfg.clone(),
            _ => CfgOptions::default(),
        }
    }

    fn edition(&self) -> Edition {
        match self {
            MockFileData::Fixture(f) => {
                f.edition.as_ref().map_or(Edition::Edition2018, |v| Edition::from_str(&v).unwrap())
            }
            _ => Edition::Edition2018,
        }
    }

    fn env(&self) -> Env {
        match self {
            MockFileData::Fixture(f) => Env::from(f.env.iter()),
            _ => Env::default(),
        }
    }
}

impl From<Fixture> for MockFileData {
    fn from(fixture: Fixture) -> Self {
        Self::Fixture(fixture)
    }
}

/// Mock analysis is used in test to bootstrap an AnalysisHost/Analysis
/// from a set of in-memory files.
#[derive(Debug, Default)]
pub struct MockAnalysis {
    files: Vec<MockFileData>,
}

impl MockAnalysis {
    pub fn new() -> MockAnalysis {
        MockAnalysis::default()
    }
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
    pub fn with_files(fixture: &str) -> MockAnalysis {
        let mut res = MockAnalysis::new();
        for entry in Fixture::parse(fixture) {
            res.add_file_fixture(entry);
        }
        res
    }

    /// Same as `with_files`, but requires that a single file contains a `<|>` marker,
    /// whose position is also returned.
    pub fn with_files_and_position(fixture: &str) -> (MockAnalysis, FilePosition) {
        let mut position = None;
        let mut res = MockAnalysis::new();
        for entry in Fixture::parse(fixture) {
            if entry.text.contains(CURSOR_MARKER) {
                assert!(position.is_none(), "only one marker (<|>) per fixture is allowed");
                position = Some(res.add_file_fixture_with_position(entry));
            } else {
                res.add_file_fixture(entry);
            }
        }
        let position = position.expect("expected a marker (<|>)");
        (res, position)
    }

    pub fn add_file_fixture(&mut self, fixture: Fixture) -> FileId {
        let file_id = self.next_id();
        self.files.push(MockFileData::from(fixture));
        file_id
    }

    pub fn add_file_fixture_with_position(&mut self, mut fixture: Fixture) -> FilePosition {
        let (offset, text) = extract_offset(&fixture.text);
        fixture.text = text;
        let file_id = self.next_id();
        self.files.push(MockFileData::from(fixture));
        FilePosition { file_id, offset }
    }

    pub fn add_file(&mut self, path: &str, text: &str) -> FileId {
        let file_id = self.next_id();
        self.files.push(MockFileData::new(path.to_string(), text.to_string()));
        file_id
    }
    pub fn add_file_with_position(&mut self, path: &str, text: &str) -> FilePosition {
        let (offset, text) = extract_offset(text);
        let file_id = self.next_id();
        self.files.push(MockFileData::new(path.to_string(), text));
        FilePosition { file_id, offset }
    }
    pub fn add_file_with_range(&mut self, path: &str, text: &str) -> FileRange {
        let (range, text) = extract_range(text);
        let file_id = self.next_id();
        self.files.push(MockFileData::new(path.to_string(), text));
        FileRange { file_id, range }
    }
    pub fn id_of(&self, path: &str) -> FileId {
        let (idx, _) = self
            .files
            .iter()
            .enumerate()
            .find(|(_, data)| path == data.path())
            .expect("no file in this mock");
        FileId(idx as u32 + 1)
    }
    pub fn analysis_host(self) -> AnalysisHost {
        let mut host = AnalysisHost::default();
        let mut change = AnalysisChange::new();
        let mut file_set = FileSet::default();
        let mut crate_graph = CrateGraph::default();
        let mut root_crate = None;
        for (i, data) in self.files.into_iter().enumerate() {
            let path = data.path();
            assert!(path.starts_with('/'));
            let cfg_options = data.cfg_options();
            let file_id = FileId(i as u32 + 1);
            let edition = data.edition();
            let env = data.env();
            if path == "/lib.rs" || path == "/main.rs" {
                root_crate = Some(crate_graph.add_crate_root(
                    file_id,
                    edition,
                    None,
                    cfg_options,
                    env,
                    Default::default(),
                ));
            } else if path.ends_with("/lib.rs") {
                let base = &path[..path.len() - "/lib.rs".len()];
                let crate_name = &base[base.rfind('/').unwrap() + '/'.len_utf8()..];
                let other_crate = crate_graph.add_crate_root(
                    file_id,
                    edition,
                    Some(CrateName::new(crate_name).unwrap()),
                    cfg_options,
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
            change.change_file(file_id, Some(Arc::new(data.content().to_owned())));
        }
        change.set_crate_graph(crate_graph);
        change.set_roots(vec![SourceRoot::new_local(file_set)]);
        host.apply_change(change);
        host
    }
    pub fn analysis(self) -> Analysis {
        self.analysis_host().analysis()
    }

    fn next_id(&self) -> FileId {
        FileId((self.files.len() + 1) as u32)
    }
}

/// Creates analysis from a multi-file fixture, returns positions marked with <|>.
pub fn analysis_and_position(ra_fixture: &str) -> (Analysis, FilePosition) {
    let (mock, position) = MockAnalysis::with_files_and_position(ra_fixture);
    (mock.analysis(), position)
}

/// Creates analysis for a single file.
pub fn single_file(ra_fixture: &str) -> (Analysis, FileId) {
    let mut mock = MockAnalysis::new();
    let file_id = mock.add_file("/main.rs", ra_fixture);
    (mock.analysis(), file_id)
}

/// Creates analysis for a single file, returns position marked with <|>.
pub fn single_file_with_position(ra_fixture: &str) -> (Analysis, FilePosition) {
    let mut mock = MockAnalysis::new();
    let pos = mock.add_file_with_position("/main.rs", ra_fixture);
    (mock.analysis(), pos)
}

/// Creates analysis for a single file, returns range marked with a pair of <|>.
pub fn single_file_with_range(ra_fixture: &str) -> (Analysis, FileRange) {
    let mut mock = MockAnalysis::new();
    let pos = mock.add_file_with_range("/main.rs", ra_fixture);
    (mock.analysis(), pos)
}
