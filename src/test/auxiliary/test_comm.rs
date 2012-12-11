// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
  Minimized version of core::comm for testing. 

  Could probably be more minimal.
 */
#[legacy_exports];

use libc::size_t;

export port;
export recv;


/**
 * A communication endpoint that can receive messages
 *
 * Each port has a unique per-task identity and may not be replicated or
 * transmitted. If a port value is copied, both copies refer to the same
 * port.  Ports may be associated with multiple `chan`s.
 */
enum port<T: Send> {
    port_t(@port_ptr<T>)
}

/// Constructs a port
fn port<T: Send>() -> port<T> {
    port_t(@port_ptr(rustrt::new_port(sys::size_of::<T>() as size_t)))
}

struct port_ptr<T:Send> {
   po: *rust_port,
}

impl<T:Send> port_ptr<T> : Drop {
    fn finalize(&self) {
        unsafe {
            debug!("in the port_ptr destructor");
               do task::unkillable {
                let yield = 0u;
                let yieldp = ptr::addr_of(&yield);
                rustrt::rust_port_begin_detach(self.po, yieldp);
                if yield != 0u {
                    task::yield();
                }
                rustrt::rust_port_end_detach(self.po);

                while rustrt::rust_port_size(self.po) > 0u as size_t {
                    recv_::<T>(self.po);
                }
                rustrt::del_port(self.po);
            }
        }
    }
}

fn port_ptr<T: Send>(po: *rust_port) -> port_ptr<T> {
    debug!("in the port_ptr constructor");
    port_ptr {
        po: po
    }
}

/**
 * Receive from a port.  If no data is available on the port then the
 * task will block until data becomes available.
 */
fn recv<T: Send>(p: port<T>) -> T { recv_((**p).po) }


/// Receive on a raw port pointer
fn recv_<T: Send>(p: *rust_port) -> T {
    let yield = 0;
    let yieldp = ptr::addr_of(&yield);
    let mut res;
    res = rusti::init::<T>();
    rustrt::port_recv(ptr::addr_of(&res) as *uint, p, yieldp);

    if yield != 0 {
        // Data isn't available yet, so res has not been initialized.
        task::yield();
    } else {
        // In the absense of compiler-generated preemption points
        // this is a good place to yield
        task::yield();
    }
    move res
}


/* Implementation details */


enum rust_port {}

type port_id = int;

#[abi = "cdecl"]
extern mod rustrt {
    #[legacy_exports];

    fn new_port(unit_sz: libc::size_t) -> *rust_port;
    fn del_port(po: *rust_port);
    fn rust_port_begin_detach(po: *rust_port,
                              yield: *libc::uintptr_t);
    fn rust_port_end_detach(po: *rust_port);
    fn rust_port_size(po: *rust_port) -> libc::size_t;
    fn port_recv(dptr: *uint, po: *rust_port,
                 yield: *libc::uintptr_t);
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn init<T>() -> T;
}


