// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::*;
use util::{compile_crate, note};

/// A crate is a unit of Rust code to be compiled into a binary or library
pub struct Crate {
    file: ~str,
    flags: ~[~str],
    cfgs: ~[~str]
}

pub struct Listener {
    cmd: ~str,
    cb: fn~()
}

pub fn run(listeners: ~[Listener]) {
    io::println(src_dir().to_str());
    io::println(work_dir().to_str());

    let cmd = os::args()[1];

    for listeners.each |listener| {
        if listener.cmd == cmd {
            (listener.cb)();
        }
    }
}

pub impl Crate {
   fn flag(flag: ~str) -> Crate {
        Crate {
            flags: vec::append(copy self.flags, ~[flag]),
            .. copy self
        }
    }

    fn flags(flags: ~[~str]) -> Crate {
        Crate {
            flags: vec::append(copy self.flags, flags),
            .. copy self
        }
    }

   fn cfg(cfg: ~str) -> Crate {
        Crate {
            cfgs: vec::append(copy self.cfgs, ~[cfg]),
            .. copy self
        }
    }

    fn cfgs(cfgs: ~[~str]) -> Crate {
        Crate {
            cfgs: vec::append(copy self.cfgs, cfgs),
            .. copy self
        }
    }
}

/// Create a crate target from a source file
pub fn Crate(file: ~str) -> Crate {
    Crate {
        file: file,
        flags: ~[],
        cfgs: ~[]
    }
}

/** 
 * Get the working directory of the package script.
 * Assumes that the package script has been compiled
 * in is the working directory.
 */
fn work_dir() -> Path {
    os::self_exe_path().get()
}

/**
 * Get the source directory of the package (i.e.
 * where the crates are located). Assumes
 * that the cwd is changed to it before
 * running this executable.
 */
fn src_dir() -> Path {
    os::getcwd()
}

pub fn args() -> ~[~str] {
    let mut args = os::args();

    args.shift();
    args.shift();

    args
}

/// Build a set of crates, should be called once
pub fn build(crates: ~[Crate]) -> bool {
    let dir = src_dir();
    let work_dir = work_dir();
    let mut success = true;

    for crates.each |&crate| {
        let path = &dir.push_rel(&Path(crate.file)).normalize();

        note(fmt!("compiling %s", path.to_str()));

        success = compile_crate(path, &work_dir, crate.flags, crate.cfgs,
                                false, false);

        if !success { break; }
    }

    os::set_exit_status(101);

    success
}

pub mod util {
    // TODO: utilities for working with things like autotools
}
