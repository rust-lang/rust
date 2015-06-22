// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Invoke the external tools to download and unpack the stage0
//! snapshot. This is done in an background thread so that it does
//! not block the current compilation process.

use std::fs::create_dir_all;
use std::process::Command;
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use build_state::*;
use configure::ConfigArgs;
use log::Tee;

pub fn download_stage0_snapshot(args : &ConfigArgs)
                                -> Receiver<BuildState<()>> {
    let (tx, rx) = channel();
    if args.use_local_rustc() || args.no_bootstrap() {
        tx.send(continue_build()).unwrap();
    } else {
        let build_triple = args.build_triple().clone();
        let host_triple = args.host_triple().clone();
        let logger = args.get_logger(&host_triple, "download_stage0_snapshot");
        let rustc_root_dir = args.rustc_root_dir();
        let build_dir = args.toplevel_build_dir();
        let script = args.src_dir().join("etc").join("get-snapshot.py");
        println!("Downloading stage0 snapshot in the background...");
        let _ = create_dir_all(args.target_build_dir(&build_triple)
                               .join("stage0").join("bin"));
        let _ = create_dir_all(build_dir.join("dl"));
        thread::spawn(move || {
            tx.send(Command::new("python")
                    .arg(&script)
                    .arg(build_triple)
                    .env("CFG_SRC_DIR", &rustc_root_dir)
                    .current_dir(&build_dir)
                    .tee(&logger)).unwrap();
        });
    }
    rx
}
