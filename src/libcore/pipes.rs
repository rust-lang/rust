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
    mut blocked_task: option<*rust_task>,
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

type rust_task = libc::c_void;

native mod rustrt {
    #[rust_stack]
    fn rust_get_task() -> *rust_task;

    #[rust_stack]
    fn task_clear_event_reject(task: *rust_task);

    fn task_wait_event(this: *rust_task) -> *libc::c_void;
    fn task_signal_event(target: *rust_task, event: *libc::c_void);
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
    let p_ = p.unwrap();
    let p = unsafe { uniquify(p_) };
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
        #debug("waking up task for %?", p_);
        alt p.blocked_task {
          some(task) {
            rustrt::task_signal_event(task, p_ as *libc::c_void);
          }
          none { fail "blocked packet has no task" }
        }

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
    let p_ = p.unwrap();
    let p = unsafe { uniquify(p_) };
    let this = rustrt::rust_get_task();
    rustrt::task_clear_event_reject(this);
    p.blocked_task = some(this);
    loop {
        let old_state = swap_state_acq((*p).state,
                                       blocked);
        #debug("%?", old_state);
        alt old_state {
          empty {
            #debug("no data available on %?, going to sleep.", p_);
            rustrt::task_wait_event(this);
            #debug("woke up, p.state = %?", p.state);
            if p.state == full {
                let mut payload = none;
                payload <-> (*p).payload;
                p.state = terminated;
                ret some(option::unwrap(payload))
            }
          }
          blocked { fail "blocking on already blocked packet" }
          full {
            let mut payload = none;
            payload <-> (*p).payload;
            p.state = terminated;
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
