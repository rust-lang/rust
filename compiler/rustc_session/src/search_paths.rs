use std::path::{Path, PathBuf};

use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_target::spec::TargetTriple;

use crate::EarlyDiagCtxt;
use crate::filesearch::make_target_lib_path;

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
/// This type augments the `PathBuf` with an `String` containing the
/// `PathBuf`'s filename. The prefix and suffix checking is much faster on the
/// `String` than the `PathBuf`. (The filename must be valid UTF-8. If it's
/// not, the entry should be skipped, because all Rust output files are valid
/// UTF-8, and so a non-UTF-8 filename couldn't be one we're looking for.)
#[derive(Clone, Debug)]
pub struct SearchPathFile {
    pub path: PathBuf,
    pub file_name_str: String,
}

#[derive(PartialEq, Clone, Copy, Debug, Hash, Eq, Encodable, Decodable, HashStable_Generic)]
pub enum PathKind {
    Native,
    Crate,
    Dependency,
    Framework,
    ExternFlag,
    All,
}

impl PathKind {
    pub fn matches(&self, kind: PathKind) -> bool {
        match (self, kind) {
            (PathKind::All, _) | (_, PathKind::All) => true,
            _ => *self == kind,
        }
    }
}

impl SearchPath {
    pub fn from_cli_opt(
        sysroot: &Path,
        triple: &TargetTriple,
        early_dcx: &EarlyDiagCtxt,
        path: &str,
        is_unstable_enabled: bool,
    ) -> Self {
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
        let dir = match path.strip_prefix("@RUSTC_BUILTIN") {
            Some(stripped) => {
                if !is_unstable_enabled {
                    #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
                    early_dcx.early_fatal(
                        "the `-Z unstable-options` flag must also be passed to \
                         enable the use of `@RUSTC_BUILTIN`",
                    );
                }

                make_target_lib_path(sysroot, triple.triple()).join("builtin").join(stripped)
            }
            None => PathBuf::from(path),
        };
        if dir.as_os_str().is_empty() {
            #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
            early_dcx.early_fatal("empty search path given via `-L`");
        }

        Self::new(kind, dir)
    }

    pub fn from_sysroot_and_triple(sysroot: &Path, triple: &str) -> Self {
        Self::new(PathKind::All, make_target_lib_path(sysroot, triple))
    }

    pub fn new(kind: PathKind, dir: PathBuf) -> Self {
        // Get the files within the directory.
        let files = match std::fs::read_dir(&dir) {
            Ok(files) => files
                .filter_map(|e| {
                    e.ok().and_then(|e| {
                        e.file_name().to_str().map(|s| SearchPathFile {
                            path: e.path(),
                            file_name_str: s.to_string(),
                        })
                    })
                })
                .collect::<Vec<_>>(),
            Err(..) => vec![],
        };

        SearchPath { kind, dir, files }
    }
}
