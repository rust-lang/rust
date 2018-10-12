// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test ignores some platforms as the particular extension trait used
// to demonstrate the issue is only available on unix. This is fine as
// the fix to suggested paths is not platform-dependent and will apply on
// these platforms also.

// ignore-windows
// ignore-cloudabi
// ignore-emscripten

use std::process::Command;
// use std::os::unix::process::CommandExt;

fn main() {
    Command::new("echo").arg("hello").exec();
}
