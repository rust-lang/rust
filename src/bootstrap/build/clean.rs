// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fs;
use std::path::Path;

use build::Build;

pub fn clean(build: &Build) {
    for host in build.config.host.iter() {

        let out = build.out.join(host);

        rm_rf(build, &out.join("compiler-rt"));
        rm_rf(build, &out.join("doc"));

        for stage in 0..4 {
            rm_rf(build, &out.join(format!("stage{}", stage)));
            rm_rf(build, &out.join(format!("stage{}-std", stage)));
            rm_rf(build, &out.join(format!("stage{}-rustc", stage)));
            rm_rf(build, &out.join(format!("stage{}-tools", stage)));
            rm_rf(build, &out.join(format!("stage{}-test", stage)));
        }
    }
}

fn rm_rf(build: &Build, path: &Path) {
    if path.exists() {
        build.verbose(&format!("removing `{}`", path.display()));
        t!(fs::remove_dir_all(path));
    }
}
