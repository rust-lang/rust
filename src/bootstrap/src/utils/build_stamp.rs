use std::{
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub struct BuildStamp {
    path: PathBuf,
    stamp: String,
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

    pub fn with_stamp(mut self, stamp: String) -> Self {
        self.stamp = stamp;
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

    pub fn remove(self) -> io::Result<()> {
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
