// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate gcc;

fn main() {
    println!("cargo:rustc-cfg=cargobuild");
    let mut cfg = gcc::Config::new();
    cfg.file("../rt/hoedown/src/autolink.c")
       .file("../rt/hoedown/src/buffer.c")
       .file("../rt/hoedown/src/document.c")
       .file("../rt/hoedown/src/escape.c")
       .file("../rt/hoedown/src/html.c")
       .file("../rt/hoedown/src/html_blocks.c")
       .file("../rt/hoedown/src/html_smartypants.c")
       .file("../rt/hoedown/src/stack.c")
       .file("../rt/hoedown/src/version.c")
       .include("../rt/hoedown/src")
       .compile("libhoedown.a");
}
