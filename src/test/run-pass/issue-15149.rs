// no-prefer-dynamic

// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice::SliceExt;
use std::old_io::{Command, fs, USER_RWX};
use std::os;
use std::old_path::BytesContainer;
use std::rand::random;

fn main() {
    // If we're the child, make sure we were invoked correctly
    let args = os::args();
    if args.len() > 1 && args[1] == "child" {
        // FIXME: This should check the whole `args[0]` instead of just
        // checking that it ends_with the executable name. This
        // is needed because of Windows, which has a different behavior.
        // See #15149 for more info.
        return assert!(args[0].ends_with(&format!("mytest{}", os::consts::EXE_SUFFIX)[]));
    }

    test();
}

fn test() {
    // If we're the parent, copy our own binary to a new directory.
    let my_path = os::self_exe_name().unwrap();
    let my_dir  = my_path.dir_path();

    let random_u32: u32 = random();
    let child_dir = Path::new(my_dir.join(format!("issue-15149-child-{}",
                                                  random_u32)));
    fs::mkdir(&child_dir, USER_RWX).unwrap();

    let child_path = child_dir.join(format!("mytest{}",
                                            os::consts::EXE_SUFFIX));
    fs::copy(&my_path, &child_path).unwrap();

    // Append the new directory to our own PATH.
    let mut path = os::split_paths(os::getenv("PATH").unwrap_or(String::new()));
    path.push(child_dir.clone());
    let path = os::join_paths(&path).unwrap();

    let child_output = Command::new("mytest").env("PATH", path)
                                             .arg("child")
                                             .output().unwrap();

    assert!(child_output.status.success(),
            format!("child assertion failed\n child stdout:\n {}\n child stderr:\n {}",
                    child_output.output.container_as_str().unwrap(),
                    child_output.error.container_as_str().unwrap()));

    fs::rmdir_recursive(&child_dir).unwrap();

}
