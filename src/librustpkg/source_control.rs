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
use version::*;

/// For a local git repo
pub fn git_clone(source: &Path, target: &Path, v: &Version) {
    assert!(os::path_is_dir(source));
    assert!(is_git_dir(source));
    if !os::path_exists(target) {
        let version_args = match v {
            &ExactRevision(ref s) => ~[~"--branch", s.to_owned()],
            _ => ~[]
        };
        debug!("Running: git clone %s %s %s", version_args.to_str(), source.to_str(),
               target.to_str());
        let outp = run::process_output("git", ~[~"clone"] + version_args +
                                       ~[source.to_str(), target.to_str()]);
        if outp.status != 0 {
            io::println(str::from_bytes_owned(outp.output.clone()));
            io::println(str::from_bytes_owned(outp.error));
            fail!("Couldn't `git clone` %s", source.to_str());
        }
    }
    else {
        // Pull changes
        debug!("Running: git --work-tree=%s --git-dir=%s pull --no-edit %s",
               target.to_str(), target.push(".git").to_str(), source.to_str());
        let outp = run::process_output("git", [fmt!("--work-tree=%s", target.to_str()),
                                               fmt!("--git-dir=%s", target.push(".git").to_str()),
                                               ~"pull", ~"--no-edit", source.to_str()]);
        assert!(outp.status == 0);
    }
}

pub fn is_git_dir(p: &Path) -> bool {
    os::path_is_dir(&p.push(".git"))
}
