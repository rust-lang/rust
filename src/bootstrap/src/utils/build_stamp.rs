use std::path::{Path, PathBuf};
use std::{fs, io};

use crate::core::builder::Builder;
use crate::core::config::TargetSelection;
use crate::{Compiler, Mode};

#[derive(Clone)]
pub struct BuildStamp {
    path: PathBuf,
    pub(crate) stamp: String,
}

impl From<BuildStamp> for PathBuf {
    fn from(value: BuildStamp) -> Self {
        value.path
    }
}

impl AsRef<Path> for BuildStamp {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl BuildStamp {
    pub fn new(dir: &Path) -> Self {
        Self { path: dir.join(".stamp"), stamp: String::new() }
    }

    pub fn with_stamp<S: ToString>(mut self, stamp: S) -> Self {
        self.stamp = stamp.to_string();
        self
    }

    pub fn with_prefix(mut self, prefix: &str) -> Self {
        assert!(
            !prefix.starts_with('.') && !prefix.ends_with('.'),
            "prefix can not start or end with '.'"
        );

        let stamp_filename = self.path.components().last().unwrap().as_os_str().to_str().unwrap();
        let stamp_filename = stamp_filename.strip_prefix('.').unwrap_or(stamp_filename);
        self.path.set_file_name(format!(".{prefix}-{stamp_filename}"));

        self
    }

    pub fn remove(&self) -> io::Result<()> {
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    pub fn write(&self) -> io::Result<()> {
        fs::write(&self.path, &self.stamp)
    }

    pub fn is_up_to_date(&self) -> bool {
        match fs::read(&self.path) {
            Ok(h) => self.stamp.as_bytes() == h.as_slice(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(e) => {
                panic!("failed to read stamp file `{}`: {}", self.path.display(), e);
            }
        }
    }
}

/// Cargo's output path for librustc_codegen_llvm in a given stage, compiled by a particular
/// compiler for the specified target and backend.
pub fn codegen_backend_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
    backend: &str,
) -> BuildStamp {
    BuildStamp::new(&builder.cargo_out(compiler, Mode::Codegen, target))
        .with_prefix(&format!("librustc_codegen_{backend}"))
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
pub fn libstd_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
) -> BuildStamp {
    BuildStamp::new(&builder.cargo_out(compiler, Mode::Std, target)).with_prefix("libstd")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn librustc_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
) -> BuildStamp {
    BuildStamp::new(&builder.cargo_out(compiler, Mode::Rustc, target)).with_prefix("librustc")
}
