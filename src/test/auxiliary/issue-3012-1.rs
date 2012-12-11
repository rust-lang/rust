// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name="socketlib", vers="0.0")];
#[crate_type = "lib"];
#[legacy_exports];

mod socket {
    #[legacy_exports];

export socket_handle;

struct socket_handle {
    sockfd: libc::c_int,
}

impl socket_handle : Drop {
    fn finalize(&self) {
        /* c::close(self.sockfd); */
    }
}

    fn socket_handle(x: libc::c_int) -> socket_handle {
        socket_handle {
            sockfd: x
        }
    }

}
