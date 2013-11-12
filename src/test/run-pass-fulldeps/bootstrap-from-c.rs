// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test exercises bootstrapping the runtime from C to make sure that this
// continues to work. The runtime is expected to have all local I/O services
// that one would normally expect.

#[feature(managed_boxes)];

extern mod extra;
extern mod rustc;
extern mod syntax;

use extra::glob;
use extra::tempfile::TempDir;
use rustc::back::link;
use rustc::driver::driver;
use rustc::driver::session;
use std::os;
use std::rt::io::fs;
use std::run;
use std::str;
use syntax::diagnostic;

fn main() {
    // Sketchily figure out where our sysroot is
    let mut sysroot = Path::new(os::self_exe_path().unwrap());
    sysroot.pop();
    sysroot.pop();
    sysroot.push("stage2");

    // Give ourselves a little workspace
    let d = TempDir::new("mytest").unwrap();
    let d = d.path();

    // Figure out where we are
    let mut me = Path::new(file!());
    me.pop();
    let srcfile = me.join("bootstrap-from-c/lib.rs");
    let cfile = me.join("bootstrap-from-c/main.c");

    // Compile the rust crate
    let options = @session::options {
        maybe_sysroot: Some(@sysroot),
        debugging_opts: session::gen_crate_map,
        ..(*session::basic_options()).clone()
    };
    let diagnostic = @diagnostic::DefaultEmitter as @diagnostic::Emitter;
    let session = driver::build_session(options, diagnostic);
    driver::compile_input(session, ~[], &driver::file_input(srcfile),
                          &Some(d.clone()), &None);

    // Copy the C source into place
    let cdst = d.join("main.c");
    let exe = d.join("out" + os::consts::EXE_SUFFIX);
    fs::copy(&cfile, &cdst);

    // Figure out where we put the dynamic library
    let dll = os::dll_filename("boot-*");
    let dll = glob::glob(d.as_str().unwrap() + "/" + dll).next().unwrap();

    // Compile the c program with all the appropriate arguments. We're compiling
    // the cfile and the library together, and may have to do some linker rpath
    // magic to make sure that the dll can get found when the executable is
    // running.
    let cc = link::get_cc_prog(session);
    let mut cc_args = session.targ_cfg.target_strs.cc_args.clone();
    cc_args.push_all([~"-o", exe.as_str().unwrap().to_owned()]);
    cc_args.push(cdst.as_str().unwrap().to_owned());
    cc_args.push(dll.as_str().unwrap().to_owned());
    if cfg!(target_os = "macos") {
        cc_args.push("-Wl,-rpath," + d.as_str().unwrap().to_owned());
    }
    let status = run::process_status(cc, cc_args);
    assert!(status.success());

    // Finally, run the program and make sure that it tells us hello.
    let res = run::process_output(exe.as_str().unwrap().to_owned(), []);
    assert!(res.status.success());
    assert_eq!(str::from_utf8(res.output), ~"hello\n");
}
