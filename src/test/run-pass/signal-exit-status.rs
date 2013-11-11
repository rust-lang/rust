// copyright 2013 the rust project developers. see the copyright
// file at the top-level directory of this distribution and at
// http://rust-lang.org/copyright.
//
// licensed under the apache license, version 2.0 <license-apache or
// http://www.apache.org/licenses/license-2.0> or the mit license
// <license-mit or http://opensource.org/licenses/mit>, at your
// option. this file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

use std::{os, run};
use std::io::process;

fn main() {
    let args = os::args();
    if args.len() >= 2 && args[1] == ~"signal" {
        // Raise a segfault.
        unsafe { *(0 as *mut int) = 0; }
    } else {
        let status = run::process_status(args[0], [~"signal"]);
        match status {
            process::ExitSignal(_) => {},
            _ => fail!("invalid termination (was not signalled): {:?}", status)
        }
    }
}

