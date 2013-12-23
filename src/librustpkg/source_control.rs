// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Utils for working with version control repositories. Just git right now.

use std::{run, str};
use std::run::{ProcessOutput, ProcessOptions, Process};
use std::io::fs;
use extra::tempfile::TempDir;
use version::*;
use path_util::chmod_read_only;

/// Attempts to clone `source`, a local git repository, into `target`, a local
/// directory that doesn't exist.
/// Returns `DirToUse(p)` if the clone fails, where `p` is a newly created temporary
/// directory (that the callee may use, for example, to check out remote sources into).
/// Returns `CheckedOutSources` if the clone succeeded.
pub fn safe_git_clone(source: &Path, v: &Version, target: &Path) -> CloneResult {
    if source.exists() {
        debug!("{} exists locally! Cloning it into {}",
                source.display(), target.display());
        // Ok to use target here; we know it will succeed
        assert!(source.is_dir());
        assert!(is_git_dir(source));

        if !target.exists() {
            debug!("Running: git clone {} {}", source.display(), target.display());
            // FIXME (#9639): This needs to handle non-utf8 paths
            let opt_outp = run::process_output("git", [~"clone",
                                                       source.as_str().unwrap().to_owned(),
                                                       target.as_str().unwrap().to_owned()]);
            let outp = opt_outp.expect("Failed to exec `git`");
            if !outp.status.success() {
                println!("{}", str::from_utf8_owned(outp.output.clone()).unwrap());
                println!("{}", str::from_utf8_owned(outp.error).unwrap());
                return DirToUse(target.clone());
            }
            else {
                match v {
                    &ExactRevision(ref s) => {
                        let git_dir = target.join(".git");
                        debug!("`Running: git --work-tree={} --git-dir={} checkout {}",
                                *s, target.display(), git_dir.display());
                        // FIXME (#9639: This needs to handle non-utf8 paths
                        let outp = run::process_output("git",
                            [format!("--work-tree={}", target.as_str().unwrap().to_owned()),
                             format!("--git-dir={}", git_dir.as_str().unwrap().to_owned()),
                             ~"checkout", format!("{}", *s)]).expect("Failed to exec `git`");
                        if !outp.status.success() {
                            println!("{}", str::from_utf8_owned(outp.output.clone()).unwrap());
                            println!("{}", str::from_utf8_owned(outp.error).unwrap());
                            return DirToUse(target.clone());
                        }
                    }
                    _ => ()
                }
            }
        } else {
            // Check that no version was specified. There's no reason to not handle the
            // case where a version was requested, but I haven't implemented it.
            assert!(*v == NoVersion);
            let git_dir = target.join(".git");
            debug!("Running: git --work-tree={} --git-dir={} pull --no-edit {}",
                    target.display(), git_dir.display(), source.display());
            // FIXME (#9639: This needs to handle non-utf8 paths
            let args = [format!("--work-tree={}", target.as_str().unwrap().to_owned()),
                        format!("--git-dir={}", git_dir.as_str().unwrap().to_owned()),
                        ~"pull", ~"--no-edit", source.as_str().unwrap().to_owned()];
            let opt_outp = run::process_output("git", args);
            let outp = opt_outp.expect("Failed to exec `git`");
            assert!(outp.status.success());
        }
        CheckedOutSources
    } else {
        use conditions::failed_to_create_temp_dir::cond;

        let scratch_dir = TempDir::new("rustpkg");
        let clone_target = match scratch_dir {
            Some(d) => d.unwrap().join("rustpkg_temp"),
            None    => cond.raise(~"Failed to create temporary directory for fetching git sources")
        };

        DirToUse(clone_target)
    }
}

pub enum CloneResult {
    DirToUse(Path), // Created this empty directory to use as the temp dir for git
    CheckedOutSources // Successfully checked sources out into the given target dir
}

pub fn make_read_only(target: &Path) {
    // Now, make all the files in the target dir read-only
    for p in fs::walk_dir(target) {
        if !p.is_dir() {
            assert!(chmod_read_only(&p));
        }
    }
}

/// Source can be either a URL or a local file path.
pub fn git_clone_url(source: &str, target: &Path, v: &Version) {
    use conditions::git_checkout_failed::cond;

    // FIXME (#9639): This needs to handle non-utf8 paths
    let opt_outp = run::process_output("git", [~"clone", source.to_owned(),
                                               target.as_str().unwrap().to_owned()]);
    let outp = opt_outp.expect("Failed to exec `git`");
    if !outp.status.success() {
         debug!("{}", str::from_utf8_owned(outp.output.clone()).unwrap());
         debug!("{}", str::from_utf8_owned(outp.error).unwrap());
         cond.raise((source.to_owned(), target.clone()))
    }
    else {
        match v {
            &ExactRevision(ref s) | &Tagged(ref s) => {
                    let opt_outp = process_output_in_cwd("git", [~"checkout", s.to_owned()],
                                                         target);
                    let outp = opt_outp.expect("Failed to exec `git`");
                    if !outp.status.success() {
                        debug!("{}", str::from_utf8_owned(outp.output.clone()).unwrap());
                        debug!("{}", str::from_utf8_owned(outp.error).unwrap());
                        cond.raise((source.to_owned(), target.clone()))
                    }
            }
            _ => ()
        }
    }
}

fn process_output_in_cwd(prog: &str, args: &[~str], cwd: &Path) -> Option<ProcessOutput> {
    let mut opt_prog = Process::new(prog, args, ProcessOptions{ dir: Some(cwd)
                                    ,..ProcessOptions::new()});
    match opt_prog {
        Some(ref mut prog) => Some(prog.finish_with_output()),
        None => None
    }
}

pub fn is_git_dir(p: &Path) -> bool {
    p.join(".git").is_dir()
}
