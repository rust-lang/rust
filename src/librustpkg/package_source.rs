// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use target::*;
use package_id::PkgId;
use std::io;
use std::io::fs;
use std::os;
use context::*;
use crate::Crate;
use messages::*;
use source_control::{safe_git_clone, git_clone_url, DirToUse, CheckedOutSources};
use source_control::{is_git_dir, make_read_only};
use path_util::{find_dir_using_rust_path_hack, make_dir_rwx_recursive, default_workspace};
use path_util::{target_build_dir, versionize, dir_has_crate_file};
use util::{compile_crate, DepMap};
use workcache_support;
use workcache_support::{digest_only_date, digest_file_with_date, crate_tag};
use extra::workcache;
use extra::treemap::TreeMap;

use rustc::driver::session;

// An enumeration of the unpacked source of a package workspace.
// This contains a list of files found in the source workspace.
#[deriving(Clone)]
pub struct PkgSrc {
    /// Root of where the package source code lives
    source_workspace: Path,
    /// If build_in_destination is true, temporary results should
    /// go in the build/ subdirectory of the destination workspace.
    /// (Otherwise, they go in the build/ subdirectory of the
    /// source workspace.) This happens if the "RUST_PATH hack" is
    /// in effect, or if sources were fetched from a remote
    /// repository.
    build_in_destination: bool,
    /// Where to install the results. May or may not be the same
    /// as source_workspace
    destination_workspace: Path,
    // Directory to start looking in for packages -- normally
    // this is workspace/src/id but it may be just workspace
    start_dir: Path,
    id: PkgId,
    libs: ~[Crate],
    mains: ~[Crate],
    tests: ~[Crate],
    benchs: ~[Crate],
}

pub enum BuildSort { InPlace, Discovered }

impl ToStr for PkgSrc {
    fn to_str(&self) -> ~str {
        format!("Package ID {} in start dir {} [workspaces = {} -> {}]",
                self.id.to_str(),
                self.start_dir.display(),
                self.source_workspace.display(),
                self.destination_workspace.display())
    }
}
condition! {
    // #6009: should this be pub or not, when #8215 is fixed?
    build_err: (~str) -> ~str;
}

// Package source files can be in a workspace
// subdirectory, or in a source control repository
enum StartDir { InWorkspace(Path), InRepo(Path) }

impl ToStr for StartDir {
    fn to_str(&self) -> ~str {
        match *self {
            InWorkspace(ref p) => format!("workspace({})", p.display()),
            InRepo(ref p)      => format!("repository({})", p.display())
        }
    }
}

impl PkgSrc {

    pub fn new(mut source_workspace: Path,
               mut destination_workspace: Path,
               use_rust_path_hack: bool,
               id: PkgId) -> PkgSrc {
        use conditions::nonexistent_package::cond;

        debug!("Checking package source for package ID {}, \
                workspace = {} -> {}, use_rust_path_hack = {:?}",
                id.to_str(),
                source_workspace.display(),
                destination_workspace.display(),
                use_rust_path_hack);

        // Possible locations to look for sources
        let mut paths_to_search = ~[];
        // Possible places to put the checked-out sources (with or without version)
        let mut possible_cache_dirs = ~[];
        // The directory in which to put build products for this package
        let where_to_build = target_build_dir(&source_workspace);

        if use_rust_path_hack {
            paths_to_search.push(InWorkspace(source_workspace.clone()));
        } else {
            // We search for sources under both src/ and build/ , because build/ is where
            // automatically-checked-out sources go.

            paths_to_search.push(
                InWorkspace(source_workspace.join("src").
                            join(id.path.clone().dir_path()).
                            join(format!("{}-{}",
                                         id.short_name.clone(),
                                         id.version.to_str().clone()))));
            paths_to_search.push(InWorkspace(source_workspace.join("src").join(id.path.clone())));
            // Try it without the src/ too, in the case of a local git repo
            paths_to_search.push(InRepo(source_workspace.join(id.path.clone().dir_path()).
                                        join(format!("{}-{}",
                                                     id.short_name.clone(),
                                                     id.version.to_str().clone()))));
            paths_to_search.push(InRepo(source_workspace.join(id.path.clone())));


            // Search the CWD
            let cwd = os::getcwd();
            paths_to_search.push(if dir_has_crate_file(&cwd) {
                    InRepo(cwd) } else { InWorkspace(cwd) });

            // Search the build dir too, since that's where automatically-downloaded sources go
            let in_build_dir =
                where_to_build.join("src").
                               join(id.path.clone().dir_path()).join(
                                          format!("{}-{}",
                                                  id.short_name.clone(),
                                                  id.version.to_str()));
            paths_to_search.push(InWorkspace(in_build_dir.clone()));
            possible_cache_dirs.push(in_build_dir);
            let in_build_dir = where_to_build.join("src").join(id.path.clone());
            paths_to_search.push(InWorkspace(in_build_dir.clone()));
            possible_cache_dirs.push(in_build_dir);

        }

        debug!("Searching for sources in the following locations: {:?}",
               paths_to_search.map(|p| p.to_str()).connect(":"));

        let dir_containing_sources_opt =
            paths_to_search.iter().find(|&d| match d {
                &InWorkspace(ref d) | &InRepo(ref d) => d.is_dir() && dir_has_crate_file(d) });

        // See the comments on the definition of PkgSrc
        let mut build_in_destination = use_rust_path_hack;

        let mut start_dir = None;
        // Used only if the sources are found in a local git repository
        let mut local_git_dir = None;
        // If we determine that the sources are findable without git
        // this gets set to false
        let mut use_git = true;
        // Now we determine the top-level directory that contains the sources for this
        // package. Normally we would expect it to be <source-workspace>/src/<id.short_name>,
        // but there are some special cases.
        match dir_containing_sources_opt {
            Some(top_dir) => {
                // If this is a local git repository, we have to check it
                // out into a workspace
                match *top_dir {
                    InWorkspace(ref d) => {
                        use_git = false;
                        start_dir = Some(d.clone());
                    }
                    InRepo(ref repo_dir) if is_git_dir(repo_dir) => {
                        local_git_dir = Some(repo_dir.clone());
                    }
                    InRepo(_) => {} // Weird, but do nothing
                }
            }
            None => {
                // See if any of the prefixes of this package ID form a valid package ID
                // That is, is this a package ID that points into the middle of a workspace?
                for (prefix, suffix) in id.prefixes_iter() {
                    let package_id = PkgId::new(prefix.as_str().unwrap());
                    let path = where_to_build.join(&package_id.path);
                    debug!("in loop: checking if {} is a directory", path.display());
                    if path.is_dir() {
                        let ps = PkgSrc::new(source_workspace,
                                             destination_workspace,
                                             use_rust_path_hack,
                                             package_id);
                        match ps {
                            PkgSrc {
                                source_workspace: source,
                                destination_workspace: destination,
                                start_dir: start,
                                id: id, _ } => {
                                let result = PkgSrc {
                                    source_workspace: source.clone(),
                                    build_in_destination: build_in_destination,
                                    destination_workspace: destination,
                                    start_dir: start.join(&suffix),
                                    id: id,
                                    libs: ~[],
                                    mains: ~[],
                                    tests: ~[],
                                    benchs: ~[]
                                };
                                debug!("pkgsrc: Returning {}", result.to_str());
                                return result;
                            }
                        }

                    };
                }
            }
        }

        if use_git {
            // At this point, we don't know whether it's named `foo-0.2` or just `foo`
            for cache_dir in possible_cache_dirs.iter() {
                let mut target_dir_for_checkout = None;
                debug!("Calling fetch_git on {}", cache_dir.display());
                let checkout_target_dir_opt = PkgSrc::fetch_git(cache_dir,
                                                                &id,
                                                                &local_git_dir);
                for checkout_target_dir in checkout_target_dir_opt.iter() {
                    target_dir_for_checkout = Some(checkout_target_dir.clone());
                    // In this case, we put the build products in the destination
                    // workspace, because this package originated from a non-workspace.
                    build_in_destination = true;
                }
                match target_dir_for_checkout {
                    Some(ref checkout_target_dir) => {
                        if checkout_target_dir.is_ancestor_of(&id.path)
                            || checkout_target_dir.is_ancestor_of(&versionize(&id.path,
                                                                              &id.version)) {
                            // This means that we successfully checked out the git sources
                            // into `checkout_target_dir`.
                            // Now determine whether or not the source was actually a workspace
                            // (that is, whether it has a `src` subdirectory)
                            // Strip off the package ID
                            source_workspace = checkout_target_dir.clone();
                            for _ in id.path.component_iter() {
                                source_workspace.pop();
                            }
                            // Strip off the src/ part
                            match source_workspace.filename_str() {
                                None => (),
                                Some("src") => {
                                    source_workspace.pop();
                                }
                                _ => ()
                            }
                        // Strip off the build/<target-triple> part to get the workspace
                        destination_workspace = source_workspace.clone();
                        destination_workspace.pop();
                        destination_workspace.pop();
                        }
                        start_dir = Some(checkout_target_dir.clone());
                        break;
                    }
                    // In this case, the git checkout did not succeed.
                    None => {
                        debug!("With cache_dir = {}, checkout failed", cache_dir.display());
                    }
                }
            }
        }

        // If we still haven't found it yet...
        if start_dir.is_none() {
            // See if the sources are in $CWD
            let cwd = os::getcwd();
            if dir_has_crate_file(&cwd) {
                return PkgSrc {
                    // In this case, source_workspace isn't really a workspace.
                    // This data structure needs yet more refactoring.
                    source_workspace: cwd.clone(),
                    destination_workspace: default_workspace(),
                    build_in_destination: true,
                    start_dir: cwd,
                    id: id,
                    libs: ~[],
                    mains: ~[],
                    benchs: ~[],
                    tests: ~[]
                }
            } else if use_rust_path_hack {
                let dir_opt = find_dir_using_rust_path_hack(&id);
                for dir_with_sources in dir_opt.iter() {
                    start_dir = Some(dir_with_sources.clone());
                }
            }
        };

        match start_dir {
            None => {
                PkgSrc {
                    source_workspace: source_workspace.clone(),
                    build_in_destination: build_in_destination,
                    destination_workspace: destination_workspace,
                    start_dir: cond.raise((id.clone(),
                                            format!("supplied path for package {} does not \
                              exist, and couldn't interpret it as a URL fragment", id.to_str()))),
                    id: id,
                    libs: ~[],
                    mains: ~[],
                    tests: ~[],
                    benchs: ~[]
                }
            }
            Some(start_dir) => {
                if !start_dir.is_dir() {
                    cond.raise((id.clone(), format!("supplied path `{}` for package dir is a \
                                non-directory", start_dir.display())));
                }
                debug!("For package id {}, returning {}", id.to_str(), start_dir.display());
                PkgSrc {
                    source_workspace: source_workspace.clone(),
                    build_in_destination: build_in_destination,
                    destination_workspace: destination_workspace,
                    start_dir: start_dir,
                    id: id,
                    libs: ~[],
                    mains: ~[],
                    tests: ~[],
                    benchs: ~[]
                }
            }
        }
    }

    /// Try interpreting self's package id as a git repository, and try
    /// fetching it and caching it in a local directory. Return the cached directory
    /// if this was successful, None otherwise. Similarly, if the package id
    /// refers to a git repo on the local version, also check it out.
    /// (right now we only support git)
    /// If parent_dir.is_some(), then we treat the pkgid's path as relative to it
    pub fn fetch_git(local: &Path, pkgid: &PkgId, parent_dir: &Option<Path>) -> Option<Path> {
        use conditions::git_checkout_failed::cond;

        let cwd = os::getcwd();
        let pkgid_path = match parent_dir {
            &Some(ref p) => p.clone(),
            &None    => pkgid.path.clone()
        };
        debug!("Checking whether {} (path = {}) exists locally. Cwd = {}, does it? {:?}",
                pkgid.to_str(), pkgid_path.display(),
                cwd.display(),
                pkgid_path.exists());

        match safe_git_clone(&pkgid_path, &pkgid.version, local) {
            CheckedOutSources => {
                make_read_only(local);
                Some(local.clone())
            }
            DirToUse(clone_target) => {
                if pkgid.path.component_iter().nth(1).is_none() {
                    // If a non-URL, don't bother trying to fetch
                    return None;
                }

                // FIXME (#9639): This needs to handle non-utf8 paths
                let url = format!("https://{}", pkgid.path.as_str().unwrap());
                debug!("Fetching package: git clone {} {} [version={}]",
                        url, clone_target.display(), pkgid.version.to_str());

                let mut failed = false;

                do cond.trap(|_| {
                    failed = true;
                }).inside {
                    git_clone_url(url, &clone_target, &pkgid.version);
                };

                if failed {
                    return None;
                }

                // Move clone_target to local.
                // First, create all ancestor directories.
                let moved = make_dir_rwx_recursive(&local.dir_path())
                    && io::result(|| fs::rename(&clone_target, local)).is_ok();
                if moved { Some(local.clone()) }
                    else { None }
            }
        }
    }

    // If a file named "pkg.rs" in the start directory exists,
    // return the path for it. Otherwise, None
    pub fn package_script_option(&self) -> Option<Path> {
        let maybe_path = self.start_dir.join("pkg.rs");
        debug!("package_script_option: checking whether {} exists", maybe_path.display());
        if maybe_path.exists() {
            Some(maybe_path)
        } else {
            None
        }
    }

    /// True if the given path's stem is self's pkg ID's stem
    fn stem_matches(&self, p: &Path) -> bool {
        p.filestem().map_default(false, |p| { p == self.id.short_name.as_bytes() })
    }

    pub fn push_crate(cs: &mut ~[Crate], prefix: uint, p: &Path) {
        let mut it = p.component_iter().peekable();
        if prefix > 0 {
            it.nth(prefix-1); // skip elements
        }
        assert!(it.peek().is_some());
        let mut sub = Path::new(".");
        for c in it {
            sub.push(c);
        }
        debug!("Will compile crate {}", sub.display());
        cs.push(Crate::new(&sub));
    }

    /// Infers crates to build. Called only in the case where there
    /// is no custom build logic
    pub fn find_crates(&mut self) {
        self.find_crates_with_filter(|_| true);
    }

    pub fn find_crates_with_filter(&mut self, filter: &fn(&str) -> bool) {
        use conditions::missing_pkg_files::cond;

        let prefix = self.start_dir.component_iter().len();
        debug!("Matching against {}", self.id.short_name);
        for pth in fs::walk_dir(&self.start_dir) {
            let maybe_known_crate_set = match pth.filename_str() {
                Some(filename) if filter(filename) => match filename {
                    "lib.rs" => Some(&mut self.libs),
                    "main.rs" => Some(&mut self.mains),
                    "test.rs" => Some(&mut self.tests),
                    "bench.rs" => Some(&mut self.benchs),
                    _ => None
                },
                _ => None
            };

            match maybe_known_crate_set {
                Some(crate_set) => PkgSrc::push_crate(crate_set, prefix, &pth),
                None => ()
            }
        }

        let crate_sets = [&self.libs, &self.mains, &self.tests, &self.benchs];
        if crate_sets.iter().all(|crate_set| crate_set.is_empty()) {

            note("Couldn't infer any crates to build.\n\
                         Try naming a crate `main.rs`, `lib.rs`, \
                         `test.rs`, or `bench.rs`.");
            cond.raise(self.id.clone());
        }

        debug!("In {}, found {} libs, {} mains, {} tests, {} benchs",
               self.start_dir.display(),
               self.libs.len(),
               self.mains.len(),
               self.tests.len(),
               self.benchs.len())
    }

    fn build_crates(&self,
                    ctx: &BuildContext,
                    deps: &mut DepMap,
                    crates: &[Crate],
                    cfgs: &[~str],
                    what: OutputType,
                    inputs_to_discover: &[(~str, Path)]) {
        for crate in crates.iter() {
            let path = self.start_dir.join(&crate.file);
            debug!("build_crates: compiling {}", path.display());
            let cfgs = crate.cfgs + cfgs;

            do ctx.workcache_context.with_prep(crate_tag(&path)) |prep| {
                debug!("Building crate {}, declaring it as an input", path.display());
                // FIXME (#9639): This needs to handle non-utf8 paths
                prep.declare_input("file", path.as_str().unwrap(),
                                   workcache_support::digest_file_with_date(&path));
                let subpath = path.clone();
                let subcfgs = cfgs.clone();
                let subcx = ctx.clone();
                let id = self.id.clone();
                let sub_dir = self.build_workspace().clone();
                let sub_flags = crate.flags.clone();
                let sub_deps = deps.clone();
                let inputs = inputs_to_discover.map(|&(ref k, ref p)|
                                                    (k.clone(), p.as_str().unwrap().to_owned()));
                do prep.exec |exec| {
                    for &(ref kind, ref p) in inputs.iter() {
                        let pth = Path::new(p.clone());
                        exec.discover_input(*kind, *p, if *kind == ~"file" {
                                digest_file_with_date(&pth)
                            } else if *kind == ~"binary" {
                                digest_only_date(&Path::new(p.clone()))
                            } else {
                                fail!("Bad kind in build_crates")
                            });
                    }
                    debug!("Compiling crate {}; its output will be in {}",
                           subpath.display(), sub_dir.display());
                    let opt: session::OptLevel = subcx.context.rustc_flags.optimization_level;
                    let result = compile_crate(&subcx,
                                               exec,
                                               &id,
                                               &subpath,
                                               &sub_dir,
                                               &mut (sub_deps.clone()),
                                               sub_flags,
                                               subcfgs,
                                               opt,
                                               what);
                    // XXX: result is an Option<Path>. The following code did not take that
                    // into account. I'm not sure if the workcache really likes seeing the
                    // output as "Some(\"path\")". But I don't know what to do about it.
                    // FIXME (#9639): This needs to handle non-utf8 paths
                    let result = result.as_ref().map(|p|p.as_str().unwrap());
                    debug!("Result of compiling {} was {}", subpath.display(), result.to_str());
                    result.to_str()
                }
            };
        }
    }

    /// Declare all the crate files in the package source as inputs
    /// (to the package)
    pub fn declare_inputs(&self, prep: &mut workcache::Prep) {
        let to_do = ~[self.libs.clone(), self.mains.clone(),
                      self.tests.clone(), self.benchs.clone()];
        debug!("In declare inputs, self = {}", self.to_str());
        for cs in to_do.iter() {
            for c in cs.iter() {
                let path = self.start_dir.join(&c.file);
                debug!("Declaring input: {}", path.display());
                // FIXME (#9639): This needs to handle non-utf8 paths
                prep.declare_input("file", path.as_str().unwrap(),
                                   workcache_support::digest_file_with_date(&path.clone()));
            }
        }
    }

    pub fn build(&self,
                 build_context: &BuildContext,
                 // DepMap is a map from str (crate name) to (kind, name) --
                 // it tracks discovered dependencies per-crate
                 cfgs: ~[~str],
                 inputs_to_discover: &[(~str, Path)]) -> DepMap {
        let mut deps = TreeMap::new();
        let libs = self.libs.clone();
        let mains = self.mains.clone();
        let tests = self.tests.clone();
        let benchs = self.benchs.clone();
        debug!("Building libs in {}, destination = {}",
               self.source_workspace.display(),
               self.build_workspace().display());
        self.build_crates(build_context,
                          &mut deps,
                          libs,
                          cfgs,
                          Lib,
                          inputs_to_discover);
        debug!("Building mains");
        self.build_crates(build_context,
                          &mut deps,
                          mains,
                          cfgs,
                          Main,
                          inputs_to_discover);
        debug!("Building tests");
        self.build_crates(build_context,
                          &mut deps,
                          tests,
                          cfgs,
                          Test,
                          inputs_to_discover);
        debug!("Building benches");
        self.build_crates(build_context,
                          &mut deps,
                          benchs,
                          cfgs,
                          Bench,
                          inputs_to_discover);
        deps
    }

    /// Return the workspace to put temporary files in. See the comment on `PkgSrc`
    pub fn build_workspace<'a>(&'a self) -> &'a Path {
        if self.build_in_destination {
            &self.destination_workspace
        }
        else {
            &self.source_workspace
        }
    }

    /// Debugging
    pub fn dump_crates(&self) {
        let crate_sets = [&self.libs, &self.mains, &self.tests, &self.benchs];
        for crate_set in crate_sets.iter() {
            for c in crate_set.iter() {
                debug!("Built crate: {}", c.file.display())
            }
        }
    }
}
