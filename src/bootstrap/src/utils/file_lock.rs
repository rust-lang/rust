//! This module provides a `FileLock` type, serving as a cross-platform file-locking wrapper
//! for `std::fs::File`, built on top of the `fs2` crate.
//!
//! Locks are automatically released with the `Drop` trait implementation.

use std::{fs::File, io};

use fs2::FileExt;

use crate::t;

pub struct FileLock(File);

impl FileLock {
    pub fn lock(&mut self) -> io::Result<&File> {
        self.0.lock_exclusive()?;
        Ok(&self.0)
    }

    pub fn try_lock(&mut self) -> io::Result<&File> {
        self.0.try_lock_exclusive()?;
        Ok(&self.0)
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        t!(self.0.unlock());
    }
}

impl From<File> for FileLock {
    fn from(file: File) -> Self {
        Self(file)
    }
}
