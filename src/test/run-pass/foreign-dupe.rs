// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast - Somehow causes check-fast to livelock?? Probably because we're
// calling pin_task and that's having wierd side-effects.

#[abi = "cdecl"]
#[link_name = "rustrt"]
extern mod rustrt1 {
    #[legacy_exports];
    fn last_os_error() -> ~str;
}

#[abi = "cdecl"]
#[link_name = "rustrt"]
extern mod rustrt2 {
    #[legacy_exports];
    fn last_os_error() -> ~str;
}

fn main() {
    rustrt1::last_os_error();
    rustrt2::last_os_error();
}
