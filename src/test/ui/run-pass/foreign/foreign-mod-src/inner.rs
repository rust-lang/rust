// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




pub fn main() {
    let f = "Makefile";
    let s = rustrt.str_buf(f);
    let buf = libc.malloc(1024);
    let fd = libc.open(s, 0, 0);
    libc.read(fd, buf, 1024);
    libc.write(1, buf, 1024);
    libc.close(fd);
    libc.free(buf);
}
