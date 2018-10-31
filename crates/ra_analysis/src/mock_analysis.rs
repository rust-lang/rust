
use std::sync::Arc;

use relative_path::{RelativePath, RelativePathBuf};
use ra_syntax::TextUnit;
use test_utils::{extract_offset, parse_fixture, CURSOR_MARKER};

use crate::{
    AnalysisChange, Analysis, AnalysisHost, FileId, FileResolver,
};

#[derive(Debug)]
pub struct FilePosition {
    pub file_id: FileId,
    pub offset: TextUnit,
}

/// Mock analysis is used in test to bootstrap an AnalysisHost/Analysis
/// from a set of in-memory files.
#[derive(Debug, Default)]
pub struct MockAnalysis {
    files: Vec<(String, String)>,
}

impl MockAnalysis {
    pub fn new() -> MockAnalysis {
        MockAnalysis::default()
    }
    /// Creates `MockAnalysis` using a fixture data in the following format:
    ///
    /// ```notrust
    /// //- /main.rs
    /// mod foo;
    /// fn main() {}
    ///
    /// //- /foo.rs
    /// struct Baz;
    /// ```
    pub fn with_files(fixture: &str) -> MockAnalysis {
        let mut res = MockAnalysis::new();
        for entry in parse_fixture(fixture) {
            res.add_file(&entry.meta, &entry.text);
        }
        res
    }

    /// Same as `with_files`, but requires that a single file contains a `<|>` marker,
    /// whose position is also returned.
    pub fn with_files_and_position(fixture: &str) -> (MockAnalysis, FilePosition) {
        let mut position = None;
        let mut res = MockAnalysis::new();
        for entry in parse_fixture(fixture) {
            if entry.text.contains(CURSOR_MARKER) {
                assert!(position.is_none(), "only one marker (<|>) per fixture is allowed");
                position = Some(res.add_file_with_position(&entry.meta, &entry.text));
            } else {
                res.add_file(&entry.meta, &entry.text);
            }
        }
        let position = position.expect("expected a marker (<|>)");
        (res, position)
    }

    pub fn add_file(&mut self, path: &str, text: &str) -> FileId {
        let file_id = FileId((self.files.len() + 1) as u32);
        self.files.push((path.to_string(), text.to_string()));
        file_id
    }
    pub fn add_file_with_position(&mut self, path: &str, text: &str) -> FilePosition {
        let (offset, text) = extract_offset(text);
        let file_id = FileId((self.files.len() + 1) as u32);
        self.files.push((path.to_string(), text.to_string()));
        FilePosition { file_id, offset }
    }
    pub fn id_of(&self, path: &str) -> FileId {
        let (idx, _) = self.files.iter().enumerate()
            .find(|(_, (p, _text))| path == p)
            .expect("no file in this mock");
        FileId(idx as u32 + 1)
    }
    pub fn analysis_host(self) -> AnalysisHost {
        let mut host = AnalysisHost::new();
        let mut file_map = Vec::new();
        let mut change = AnalysisChange::new();
        for (id, (path, contents)) in self.files.into_iter().enumerate() {
            let file_id = FileId((id + 1) as u32);
            assert!(path.starts_with('/'));
            let path = RelativePathBuf::from_path(&path[1..]).unwrap();
            change.add_file(file_id, contents);
            file_map.push((file_id, path));
        }
        change.set_file_resolver(Arc::new(FileMap(file_map)));
        host.apply_change(change);
        host
    }
    pub fn analysis(self) -> Analysis {
        self.analysis_host().analysis()
    }
}

/// Creates analysis from a multi-file fixture, returns positions marked with <|>.
pub fn analysis_and_position(fixture: &str) -> (Analysis, FilePosition) {
    let (mock, position) = MockAnalysis::with_files_and_position(fixture);
    (mock.analysis(), position)
}

/// Creates analysis for a single file.
pub fn single_file(code: &str) -> (Analysis, FileId) {
    let mut mock = MockAnalysis::new();
    let file_id = mock.add_file("/main.rs", code);
    (mock.analysis(), file_id)
}

/// Creates analysis for a single file, returns position marked with <|>.
pub fn single_file_with_position(code: &str) -> (Analysis, FilePosition) {
    let mut mock = MockAnalysis::new();
    let pos = mock.add_file_with_position("/main.rs", code);
    (mock.analysis(), pos)
}

#[derive(Debug)]
struct FileMap(Vec<(FileId, RelativePathBuf)>);

impl FileMap {
    fn iter<'a>(&'a self) -> impl Iterator<Item = (FileId, &'a RelativePath)> + 'a {
        self.0
            .iter()
            .map(|(id, path)| (*id, path.as_relative_path()))
    }

    fn path(&self, id: FileId) -> &RelativePath {
        self.iter().find(|&(it, _)| it == id).unwrap().1
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
