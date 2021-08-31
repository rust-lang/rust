use crate::filesearch::make_target_lib_path;
use crate::{config, early_error};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SearchPath {
    pub kind: PathKind,
    pub dir: PathBuf,
    pub files: Vec<SearchPathFile>,
}

/// The obvious implementation of `SearchPath::files` is a `Vec<PathBuf>`. But
/// it is searched repeatedly by `find_library_crate`, and the searches involve
/// checking the prefix and suffix of the filename of each `PathBuf`. This is
/// doable, but very slow, because it involves calls to `file_name` and
/// `extension` that are themselves slow.
///
/// This type augments the `PathBuf` with an `Option<String>` containing the
/// `PathBuf`'s filename. The prefix and suffix checking is much faster on the
/// `Option<String>` than the `PathBuf`. (It's an `Option` because
/// `Path::file_name` can fail; if that happens then all subsequent checking
/// will also fail, which is fine.)
#[derive(Clone, Debug)]
pub struct SearchPathFile {
    pub path: PathBuf,
    pub file_name_str: Option<String>,
}

impl SearchPathFile {
    fn new(path: PathBuf) -> SearchPathFile {
        let file_name_str = path.file_name().and_then(|f| f.to_str()).map(|s| s.to_string());
        SearchPathFile { path, file_name_str }
    }
}

#[derive(PartialEq, Clone, Copy, Debug, Hash, Eq, Encodable, Decodable)]
pub enum PathKind {
    Native,
    Crate,
    Dependency,
    Framework,
    ExternFlag,
    All,
}

rustc_data_structures::impl_stable_hash_via_hash!(PathKind);

impl PathKind {
    pub fn matches(&self, kind: PathKind) -> bool {
        match (self, kind) {
            (PathKind::All, _) | (_, PathKind::All) => true,
            _ => *self == kind,
        }
    }
}

impl SearchPath {
    pub fn from_cli_opt(path: &str, output: config::ErrorOutputType) -> Self {
        let (kind, path) = if let Some(stripped) = path.strip_prefix("native=") {
            (PathKind::Native, stripped)
        } else if let Some(stripped) = path.strip_prefix("crate=") {
            (PathKind::Crate, stripped)
        } else if let Some(stripped) = path.strip_prefix("dependency=") {
            (PathKind::Dependency, stripped)
        } else if let Some(stripped) = path.strip_prefix("framework=") {
            (PathKind::Framework, stripped)
        } else if let Some(stripped) = path.strip_prefix("all=") {
            (PathKind::All, stripped)
        } else {
            (PathKind::All, path)
        };
        if path.is_empty() {
            early_error(output, "empty search path given via `-L`");
        }

        let dir = PathBuf::from(path);
        Self::new(kind, dir)
    }

    pub fn from_sysroot_and_triple(sysroot: &Path, triple: &str) -> Self {
        Self::new(PathKind::All, make_target_lib_path(sysroot, triple))
    }

    fn new(kind: PathKind, dir: PathBuf) -> Self {
        // Get the files within the directory.
        let files = match std::fs::read_dir(&dir) {
            Ok(files) => files
                .filter_map(|e| e.ok().map(|e| SearchPathFile::new(e.path())))
                .collect::<Vec<_>>(),
            Err(..) => vec![],
        };

        SearchPath { kind, dir, files }
    }
}
