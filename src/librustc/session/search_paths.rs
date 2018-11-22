// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::{Path, PathBuf};
use session::{early_error, config};
use session::filesearch::make_target_lib_path;

#[derive(Clone, Debug)]
pub struct SearchPath {
    pub kind: PathKind,
    pub dir: PathBuf,
}

#[derive(Eq, PartialEq, Clone, Copy, Debug, PartialOrd, Ord, Hash)]
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
    pub fn from_cli_opt(path: &str, output: config::ErrorOutputType) -> Self {
        let (kind, path) = if path.starts_with("native=") {
            (PathKind::Native, &path["native=".len()..])
        } else if path.starts_with("crate=") {
            (PathKind::Crate, &path["crate=".len()..])
        } else if path.starts_with("dependency=") {
            (PathKind::Dependency, &path["dependency=".len()..])
        } else if path.starts_with("framework=") {
            (PathKind::Framework, &path["framework=".len()..])
        } else if path.starts_with("all=") {
            (PathKind::All, &path["all=".len()..])
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
        SearchPath { kind, dir }
    }
}

