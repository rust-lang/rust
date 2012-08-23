/*
  Minimized version of core::comm for testing. 

  Could probably be more minimal.
 */

import libc::size_t;

export port::{};
export port;
export recv;


/**
 * A communication endpoint that can receive messages
 *
 * Each port has a unique per-task identity and may not be replicated or
 * transmitted. If a port value is copied, both copies refer to the same
 * port.  Ports may be associated with multiple `chan`s.
 */
enum port<T: send> {
    port_t(@port_ptr<T>)
}

/// Constructs a port
fn port<T: send>() -> port<T> {
    port_t(@port_ptr(rustrt::new_port(sys::size_of::<T>() as size_t)))
}

struct port_ptr<T:send> {
   let po: *rust_port;
   new(po: *rust_port) {
    debug!("in the port_ptr constructor");
    self.po = po; }
   drop unsafe {
    debug!("in the port_ptr destructor");
       do task::unkillable {
        let yield = 0u;
        let yieldp = ptr::addr_of(yield);
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


/**
 * Receive from a port.  If no data is available on the port then the
 * task will block until data becomes available.
 */
fn recv<T: send>(p: port<T>) -> T { recv_((**p).po) }


/// Receive on a raw port pointer
fn recv_<T: send>(p: *rust_port) -> T {
    let yield = 0u;
    let yieldp = ptr::addr_of(yield);
    let mut res;
    res = rusti::init::<T>();
    rustrt::port_recv(ptr::addr_of(res) as *uint, p, yieldp);

    if yield != 0u {
        // Data isn't available yet, so res has not been initialized.
        task::yield();
    } else {
        // In the absense of compiler-generated preemption points
        // this is a good place to yield
        task::yield();
    }
    return res;
}


/* Implementation details */


enum rust_port {}

type port_id = int;

#[abi = "cdecl"]
extern mod rustrt {

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
    fn init<T>() -> T;
}


