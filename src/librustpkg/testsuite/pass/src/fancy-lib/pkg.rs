// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod rustpkg;
extern mod rustc;

use std::{io, os};
use rustpkg::api;
use rustpkg::version::NoVersion;

use rustc::metadata::filesearch;

pub fn main() {
    use std::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
    let args = os::args();

// by convention, first arg is sysroot
    if args.len() < 2 {
        fail!("Package script requires a directory where rustc libraries live as the first \
               argument");
    }

    let sysroot_arg = args[1].clone();
    let sysroot = Path(sysroot_arg);
    if !os::path_exists(&sysroot) {
        fail!("Package script requires a sysroot that exists; %s doesn't", sysroot.to_str());
    }

    if args[2] != ~"install" {
        printfln!("Warning: I don't know how to %s", args[2]);
        return;
    }

    let out_path = Path("build/fancy-lib");
    if !os::path_exists(&out_path) {
        assert!(os::make_dir(&out_path, (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    }

    let file = io::file_writer(&out_path.push("generated.rs"),
                               [io::Create]).unwrap();
    file.write_str("pub fn wheeeee() { for [1, 2, 3].each() |_| { assert!(true); } }");


    debug!("api_____install_____lib, my sysroot:");
    debug!(sysroot.to_str());

    api::install_lib(@sysroot, os::getcwd(), ~"fancy-lib", Path("lib.rs"),
                     NoVersion);
}
