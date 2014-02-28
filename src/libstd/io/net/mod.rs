// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Networking I/O

pub use self::addrinfo::get_host_addresses;

pub mod addrinfo;
pub mod tcp;
pub mod udp;
pub mod ip;
// FIXME(#12093) - this should not be called unix
pub mod unix;
