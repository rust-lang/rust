
use std::sync::Arc;

use relative_path::{RelativePath, RelativePathBuf};

use crate::{
    AnalysisChange, Analysis, AnalysisHost, FileId, FileResolver,
};

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
    pub fn with_files(files: &[(&str, &str)]) -> MockAnalysis {
        let files = files.iter()
            .map(|it| (it.0.to_string(), it.1.to_string()))
            .collect();
        MockAnalysis { files }
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
