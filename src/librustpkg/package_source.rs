// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::*;
use package_id::PkgId;
use std::path::Path;
use std::os;
use context::*;
use crate::Crate;
use messages::*;
use source_control::{git_clone, git_clone_general};
use path_util::pkgid_src_in_workspace;
use util::compile_crate;

// An enumeration of the unpacked source of a package workspace.
// This contains a list of files found in the source workspace.
pub struct PkgSrc {
    root: Path, // root of where the package source code lives
    id: PkgId,
    libs: ~[Crate],
    mains: ~[Crate],
    tests: ~[Crate],
    benchs: ~[Crate],
}

condition! {
    build_err: (~str) -> ();
}

impl PkgSrc {

    pub fn new(src_dir: &Path, id: &PkgId) -> PkgSrc {
        PkgSrc {
            root: (*src_dir).clone(),
            id: (*id).clone(),
            libs: ~[],
            mains: ~[],
            tests: ~[],
            benchs: ~[]
        }
    }


    fn check_dir(&self) -> Path {
        use conditions::nonexistent_package::cond;

        debug!("Pushing onto root: %s | %s", self.id.path.to_str(), self.root.to_str());

        let dirs = pkgid_src_in_workspace(&self.id, &self.root);
        debug!("Checking dirs: %?", dirs);
        let path = dirs.iter().find(|&d| os::path_exists(d));

        let dir = match path {
            Some(d) => (*d).clone(),
            None => match self.fetch_git() {
                Some(d) => d,
                None => cond.raise((self.id.clone(), ~"supplied path for package dir does not \
                                      exist, and couldn't interpret it as a URL fragment"))
            }
        };
        if !os::path_is_dir(&dir) {
            cond.raise((self.id.clone(), ~"supplied path for package dir is a \
                                        non-directory"));
        }

        dir
    }

    /// Try interpreting self's package id as a git repository, and try
    /// fetching it and caching it in a local directory. Return the cached directory
    /// if this was successful, None otherwise. Similarly, if the package id
    /// refers to a git repo on the local version, also check it out.
    /// (right now we only support git)
    pub fn fetch_git(&self) -> Option<Path> {

        let mut local = self.root.push("src");
        local = local.push(self.id.to_str());
        // Git can't clone into a non-empty directory
        os::remove_dir_recursive(&local);

        debug!("Checking whether %s exists locally. Cwd = %s, does it? %?",
               self.id.path.to_str(),
               os::getcwd().to_str(),
               os::path_exists(&self.id.path));

        if os::path_exists(&self.id.path) {
            debug!("%s exists locally! Cloning it into %s",
                   self.id.path.to_str(), local.to_str());
            git_clone(&self.id.path, &local, &self.id.version);
            return Some(local);
        }

        let url = fmt!("https://%s", self.id.path.to_str());
        note(fmt!("Fetching package: git clone %s %s [version=%s]",
                  url, local.to_str(), self.id.version.to_str()));
        if git_clone_general(url, &local, &self.id.version) {
            Some(local)
        }
        else {
            None
        }
    }


    // If a file named "pkg.rs" in the current directory exists,
    // return the path for it. Otherwise, None
    pub fn package_script_option(&self, cwd: &Path) -> Option<Path> {
        let maybe_path = cwd.push("pkg.rs");
        if os::path_exists(&maybe_path) {
            Some(maybe_path)
        }
        else {
            None
        }
    }

    /// True if the given path's stem is self's pkg ID's stem
    fn stem_matches(&self, p: &Path) -> bool {
        p.filestem().map_default(false, |p| { p == &self.id.short_name })
    }

    fn push_crate(cs: &mut ~[Crate], prefix: uint, p: &Path) {
        assert!(p.components.len() > prefix);
        let mut sub = Path("");
        for c in p.components.slice(prefix, p.components.len()).iter() {
            sub = sub.push(*c);
        }
        debug!("found crate %s", sub.to_str());
        cs.push(Crate::new(&sub));
    }

    /// Infers crates to build. Called only in the case where there
    /// is no custom build logic
    pub fn find_crates(&mut self) {
        use conditions::missing_pkg_files::cond;

        let dir = self.check_dir();
        debug!("Called check_dir, I'm in %s", dir.to_str());
        let prefix = dir.components.len();
        debug!("Matching against %?", self.id.short_name);
        do os::walk_dir(&dir) |pth| {
            let maybe_known_crate_set = match pth.filename() {
                Some(filename) => match filename {
                    ~"lib.rs" => Some(&mut self.libs),
                    ~"main.rs" => Some(&mut self.mains),
                    ~"test.rs" => Some(&mut self.tests),
                    ~"bench.rs" => Some(&mut self.benchs),
                    _ => None
                },
                _ => None
            };

            match maybe_known_crate_set {
                Some(crate_set) => PkgSrc::push_crate(crate_set, prefix, pth),
                None => ()
            }
            true
        };

        let crate_sets = [&self.libs, &self.mains, &self.tests, &self.benchs];
        if crate_sets.iter().all(|crate_set| crate_set.is_empty()) {

            note("Couldn't infer any crates to build.\n\
                         Try naming a crate `main.rs`, `lib.rs`, \
                         `test.rs`, or `bench.rs`.");
            cond.raise(self.id.clone());
        }

        debug!("found %u libs, %u mains, %u tests, %u benchs",
               self.libs.len(),
               self.mains.len(),
               self.tests.len(),
               self.benchs.len())
    }

    fn build_crates(&self,
                    ctx: &Ctx,
                    src_dir: &Path,
                    crates: &[Crate],
                    cfgs: &[~str],
                    what: OutputType) {
        for crate in crates.iter() {
            let path = &src_dir.push_rel(&crate.file).normalize();
            note(fmt!("build_crates: compiling %s", path.to_str()));
            note(fmt!("build_crates: using as workspace %s", self.root.to_str()));

            let result = compile_crate(ctx,
                                       &self.id,
                                       path,
                                       // compile_crate wants the workspace
                                       &self.root,
                                       crate.flags,
                                       crate.cfgs + cfgs,
                                       false,
                                       what);
            if !result {
                build_err::cond.raise(fmt!("build failure on %s",
                                           path.to_str()));
            }
            debug!("Result of compiling %s was %?",
                   path.to_str(), result);
        }
    }

    pub fn build(&self, ctx: &Ctx, cfgs: ~[~str]) {
        let dir = self.check_dir();
        debug!("Building libs in %s", dir.to_str());
        self.build_crates(ctx, &dir, self.libs, cfgs, Lib);
        debug!("Building mains");
        self.build_crates(ctx, &dir, self.mains, cfgs, Main);
        debug!("Building tests");
        self.build_crates(ctx, &dir, self.tests, cfgs, Test);
        debug!("Building benches");
        self.build_crates(ctx, &dir, self.benchs, cfgs, Bench);
    }
}
