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

use std::{io, os, run, str};
use std::run::{ProcessOutput, ProcessOptions, Process};
use version::*;

/// For a local git repo
pub fn git_clone(source: &Path, target: &Path, v: &Version) {
    assert!(os::path_is_dir(source));
    assert!(is_git_dir(source));
    if !os::path_exists(target) {
        debug2!("Running: git clone {} {}", source.to_str(), target.to_str());
        let outp = run::process_output("git", [~"clone", source.to_str(), target.to_str()]);
        if outp.status != 0 {
            io::println(str::from_utf8_owned(outp.output.clone()));
            io::println(str::from_utf8_owned(outp.error));
            fail2!("Couldn't `git clone` {}", source.to_str());
        }
        else {
            match v {
                &ExactRevision(ref s) => {
                    debug2!("`Running: git --work-tree={} --git-dir={} checkout {}",
                           *s, target.to_str(), target.push(".git").to_str());
                    let outp = run::process_output("git",
                                   [format!("--work-tree={}", target.to_str()),
                                    format!("--git-dir={}", target.push(".git").to_str()),
                                    ~"checkout", format!("{}", *s)]);
                    if outp.status != 0 {
                        io::println(str::from_utf8_owned(outp.output.clone()));
                        io::println(str::from_utf8_owned(outp.error));
                        fail2!("Couldn't `git checkout {}` in {}",
                              *s, target.to_str());
                    }
                }
                _ => ()
            }
        }
    }
    else {
        // Check that no version was specified. There's no reason to not handle the
        // case where a version was requested, but I haven't implemented it.
        assert!(*v == NoVersion);
        debug2!("Running: git --work-tree={} --git-dir={} pull --no-edit {}",
               target.to_str(), target.push(".git").to_str(), source.to_str());
        let args = [format!("--work-tree={}", target.to_str()),
                    format!("--git-dir={}", target.push(".git").to_str()),
                    ~"pull", ~"--no-edit", source.to_str()];
        let outp = run::process_output("git", args);
        assert!(outp.status == 0);
    }
}

/// Source can be either a URL or a local file path.
/// true if successful
pub fn git_clone_general(source: &str, target: &Path, v: &Version) -> bool {
    let outp = run::process_output("git", [~"clone", source.to_str(), target.to_str()]);
    if outp.status != 0 {
         debug2!("{}", str::from_utf8_owned(outp.output.clone()));
         debug2!("{}", str::from_utf8_owned(outp.error));
         false
    }
    else {
        match v {
            &ExactRevision(ref s) | &Tagged(ref s) => {
                    let outp = process_output_in_cwd("git", [~"checkout", format!("{}", *s)],
                                                         target);
                    if outp.status != 0 {
                        debug2!("{}", str::from_utf8_owned(outp.output.clone()));
                        debug2!("{}", str::from_utf8_owned(outp.error));
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
