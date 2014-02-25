// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// this is the rust entry point that we're going to call.
int foo(int argc, char *argv[]);

extern void (*set_crate_map)(void *map);
extern int _rust_crate_map_toplevel;

int main(int argc, char *argv[]) {
  set_crate_map(&_rust_crate_map_toplevel);
  return foo(argc, argv);
}
