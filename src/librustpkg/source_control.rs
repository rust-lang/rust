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

use std::{os, run, str};
use std::run::{ProcessOutput, ProcessOptions, Process};
use version::*;

/// For a local git repo
pub fn git_clone(source: &Path, target: &Path, v: &Version) {
    assert!(os::path_is_dir(source));
    assert!(is_git_dir(source));
    if !os::path_exists(target) {
        debug!("Running: git clone %s %s", source.to_str(),
               target.to_str());
        assert!(git_clone_general(source.to_str(), target, v));
    }
    else {
        // Pull changes
        // Note that this ignores tags, which is probably wrong. There are no tests for
        // it, though.
        debug!("Running: git --work-tree=%s --git-dir=%s pull --no-edit %s",
               target.to_str(), target.push(".git").to_str(), source.to_str());
        let outp = run::process_output("git", [fmt!("--work-tree=%s", target.to_str()),
                                               fmt!("--git-dir=%s", target.push(".git").to_str()),
                                               ~"pull", ~"--no-edit", source.to_str()]);
        assert!(outp.status == 0);
    }
}

/// Source can be either a URL or a local file path.
/// true if successful
pub fn git_clone_general(source: &str, target: &Path, v: &Version) -> bool {
    let outp = run::process_output("git", [~"clone", source.to_str(), target.to_str()]);
    if outp.status != 0 {
         debug!(str::from_bytes_owned(outp.output.clone()));
         debug!(str::from_bytes_owned(outp.error));
         false
    }
    else {
        match v {
            &ExactRevision(ref s) | &Tagged(ref s) => {
                    let outp = process_output_in_cwd("git", [~"checkout", fmt!("%s", *s)],
                                                         target);
                    if outp.status != 0 {
                        debug!(str::from_bytes_owned(outp.output.clone()));
                        debug!(str::from_bytes_owned(outp.error));
                        false
                    }
                    else {
                        true
                    }
                }
                _ => true
            }
        }
}

fn process_output_in_cwd(prog: &str, args: &[~str], cwd: &Path) -> ProcessOutput {
    let mut prog = Process::new(prog, args, ProcessOptions{ dir: Some(cwd)
                                ,..ProcessOptions::new()});
    prog.finish_with_output()
}

pub fn is_git_dir(p: &Path) -> bool {
    os::path_is_dir(&p.push(".git"))
}
