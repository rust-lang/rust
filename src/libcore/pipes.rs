// Runtime support for pipes.

import unsafe::{forget, reinterpret_cast};

enum state {
    empty,
    full,
    blocked,
    terminated
}

type packet<T: send> = {
    mut state: state,
    mut blocked_task: option<task::task>,
    mut payload: option<T>
};

fn packet<T: send>() -> *packet<T> unsafe {
    let p: *packet<T> = unsafe::transmute(~{
        mut state: empty,
        mut blocked_task: none::<task::task>,
        mut payload: none::<T>
    });
    p
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn atomic_xchng(&dst: int, src: int) -> int;
    fn atomic_xchng_acq(&dst: int, src: int) -> int;
    fn atomic_xchng_rel(&dst: int, src: int) -> int;
}

// We should consider moving this to core::unsafe, although I
// suspect graydon would want us to use void pointers instead.
unsafe fn uniquify<T>(x: *T) -> ~T {
    unsafe { unsafe::reinterpret_cast(x) }
}

fn swap_state_acq(&dst: state, src: state) -> state {
    unsafe {
        reinterpret_cast(rusti::atomic_xchng_acq(
            *(ptr::mut_addr_of(dst) as *mut int),
            src as int))
    }
}

fn swap_state_rel(&dst: state, src: state) -> state {
    unsafe {
        reinterpret_cast(rusti::atomic_xchng_rel(
            *(ptr::mut_addr_of(dst) as *mut int),
            src as int))
    }
}

fn send<T: send>(-p: send_packet<T>, -payload: T) {
    let p = p.unwrap();
    let p = unsafe { uniquify(p) };
    assert (*p).payload == none;
    (*p).payload <- some(payload);
    let old_state = swap_state_rel((*p).state, full);
    alt old_state {
      empty {
        // Yay, fastpath.

        // The receiver will eventually clean this up.
        unsafe { forget(p); }
      }
      full { fail "duplicate send" }
      blocked {
        // TODO: once the target will actually block, tell the
        // scheduler to wake it up.

        // The receiver will eventually clean this up.
        unsafe { forget(p); }
      }
      terminated {
        // The receiver will never receive this. Rely on drop_glue
        // to clean everything up.
      }
    }
}

fn recv<T: send>(-p: recv_packet<T>) -> option<T> {
    let p = p.unwrap();
    let p = unsafe { uniquify(p) };
    loop {
        let old_state = swap_state_acq((*p).state,
                                       blocked);
        alt old_state {
          empty | blocked { task::yield(); }
          full {
            let mut payload = none;
            payload <-> (*p).payload;
            ret some(option::unwrap(payload))
          }
          terminated {
            assert old_state == terminated;
            ret none;
          }
        }
    }
}

fn sender_terminate<T: send>(p: *packet<T>) {
    let p = unsafe { uniquify(p) };
    alt swap_state_rel((*p).state, terminated) {
      empty | blocked {
        // The receiver will eventually clean up.
        unsafe { forget(p) }
      }
      full {
        // This is impossible
        fail "you dun goofed"
      }
      terminated {
        // I have to clean up, use drop_glue
      }
    }
}

fn receiver_terminate<T: send>(p: *packet<T>) {
    let p = unsafe { uniquify(p) };
    alt swap_state_rel((*p).state, terminated) {
      empty {
        // the sender will clean up
        unsafe { forget(p) }
      }
      blocked {
        // this shouldn't happen.
        fail "terminating a blocked packet"
      }
      terminated | full {
        // I have to clean up, use drop_glue
      }
    }
}

class send_packet<T: send> {
    let mut p: option<*packet<T>>;
    new(p: *packet<T>) {
        //#error("take send %?", p);
        self.p = some(p);
    }
    drop {
        //if self.p != none {
        //    #error("drop send %?", option::get(self.p));
        //}
        if self.p != none {
            let mut p = none;
            p <-> self.p;
            sender_terminate(option::unwrap(p))
        }
    }
    fn unwrap() -> *packet<T> {
        let mut p = none;
        p <-> self.p;
        option::unwrap(p)
    }
}

class recv_packet<T: send> {
    let mut p: option<*packet<T>>;
    new(p: *packet<T>) {
        //#error("take recv %?", p);
        self.p = some(p);
    }
    drop {
        //if self.p != none {
        //    #error("drop recv %?", option::get(self.p));
        //}
        if self.p != none {
            let mut p = none;
            p <-> self.p;
            receiver_terminate(option::unwrap(p))
        }
    }
    fn unwrap() -> *packet<T> {
        let mut p = none;
        p <-> self.p;
        option::unwrap(p)
    }
}

fn entangle<T: send>() -> (send_packet<T>, recv_packet<T>) {
    let p = packet();
    (send_packet(p), recv_packet(p))
}

fn spawn_service<T: send>(
    init: native fn() -> (send_packet<T>, recv_packet<T>),
    +service: fn~(+recv_packet<T>))
    -> send_packet<T>
{
    let (client, server) = init();

    // This is some nasty gymnastics required to safely move the pipe
    // into a new task.
    let server = ~mut some(server);
    do task::spawn |move service| {
        let mut server_ = none;
        server_ <-> *server;
        service(option::unwrap(server_))
    }

    client
}
