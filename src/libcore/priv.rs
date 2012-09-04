// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

#[doc(hidden)];

export chan_from_global_ptr, weaken_task;

import compare_and_swap = rustrt::rust_compare_and_swap_ptr;
import task::TaskBuilder;

#[allow(non_camel_case_types)] // runtime type
type rust_port_id = uint;

extern mod rustrt {
    fn rust_compare_and_swap_ptr(address: *libc::uintptr_t,
                                 oldval: libc::uintptr_t,
                                 newval: libc::uintptr_t) -> bool;
    fn rust_task_weaken(ch: rust_port_id);
    fn rust_task_unweaken(ch: rust_port_id);
}

type GlobalPtr = *libc::uintptr_t;

/**
 * Atomically gets a channel from a pointer to a pointer-sized memory location
 * or, if no channel exists creates and installs a new channel and sets up a
 * new task to receive from it.
 */
unsafe fn chan_from_global_ptr<T: send>(
    global: GlobalPtr,
    task_fn: fn() -> task::TaskBuilder,
    +f: fn~(comm::Port<T>)
) -> comm::Chan<T> {

    enum Msg {
        Proceed,
        Abort
    }

    log(debug,~"ENTERING chan_from_global_ptr, before is_prob_zero check");
    let is_probably_zero = *global == 0u;
    log(debug,~"after is_prob_zero check");
    if is_probably_zero {
        log(debug,~"is probably zero...");
        // There's no global channel. We must make it

        let (setup_po, setup_ch) = do task_fn().spawn_conversation
            |setup_po, setup_ch| {
            let po = comm::Port::<T>();
            let ch = comm::Chan(po);
            comm::send(setup_ch, ch);

            // Wait to hear if we are the official instance of
            // this global task
            match comm::recv::<Msg>(setup_po) {
              Proceed => f(po),
              Abort => ()
            }
        };

        log(debug,~"before setup recv..");
        // This is the proposed global channel
        let ch = comm::recv(setup_po);
        // 0 is our sentinal value. It is not a valid channel
        assert unsafe::reinterpret_cast(&ch) != 0u;

        // Install the channel
        log(debug,~"BEFORE COMPARE AND SWAP");
        let swapped = compare_and_swap(
            global, 0u, unsafe::reinterpret_cast(&ch));
        log(debug,fmt!("AFTER .. swapped? %?", swapped));

        if swapped {
            // Success!
            comm::send(setup_ch, Proceed);
            ch
        } else {
            // Somebody else got in before we did
            comm::send(setup_ch, Abort);
            unsafe::reinterpret_cast(&*global)
        }
    } else {
        log(debug, ~"global != 0");
        unsafe::reinterpret_cast(&*global)
    }
}

#[test]
fn test_from_global_chan1() {

    // This is unreadable, right?

    // The global channel
    let globchan = 0u;
    let globchanp = ptr::addr_of(globchan);

    // Create the global channel, attached to a new task
    let ch = unsafe {
        do chan_from_global_ptr(globchanp, task::task) |po| {
            let ch = comm::recv(po);
            comm::send(ch, true);
            let ch = comm::recv(po);
            comm::send(ch, true);
        }
    };
    // Talk to it
    let po = comm::Port();
    comm::send(ch, comm::Chan(po));
    assert comm::recv(po) == true;

    // This one just reuses the previous channel
    let ch = unsafe {
        do chan_from_global_ptr(globchanp, task::task) |po| {
            let ch = comm::recv(po);
            comm::send(ch, false);
        }
    };

    // Talk to the original global task
    let po = comm::Port();
    comm::send(ch, comm::Chan(po));
    assert comm::recv(po) == true;
}

#[test]
fn test_from_global_chan2() {

    for iter::repeat(100u) {
        // The global channel
        let globchan = 0u;
        let globchanp = ptr::addr_of(globchan);

        let resultpo = comm::Port();
        let resultch = comm::Chan(resultpo);

        // Spawn a bunch of tasks that all want to compete to
        // create the global channel
        for uint::range(0u, 10u) |i| {
            do task::spawn {
                let ch = unsafe {
                    do chan_from_global_ptr(
                        globchanp, task::task) |po| {

                        for uint::range(0u, 10u) |_j| {
                            let ch = comm::recv(po);
                            comm::send(ch, {i});
                        }
                    }
                };
                let po = comm::Port();
                comm::send(ch, comm::Chan(po));
                // We are The winner if our version of the
                // task was installed
                let winner = comm::recv(po);
                comm::send(resultch, winner == i);
            }
        }
        // There should be only one winner
        let mut winners = 0u;
        for uint::range(0u, 10u) |_i| {
            let res = comm::recv(resultpo);
            if res { winners += 1u };
        }
        assert winners == 1u;
    }
}

/**
 * Convert the current task to a 'weak' task temporarily
 *
 * As a weak task it will not be counted towards the runtime's set
 * of live tasks. When there are no more outstanding live (non-weak) tasks
 * the runtime will send an exit message on the provided channel.
 *
 * This function is super-unsafe. Do not use.
 *
 * # Safety notes
 *
 * * Weak tasks must either die on their own or exit upon receipt of
 *   the exit message. Failure to do so will cause the runtime to never
 *   exit
 * * Tasks must not call `weaken_task` multiple times. This will
 *   break the kernel's accounting of live tasks.
 * * Weak tasks must not be supervised. A supervised task keeps
 *   a reference to its parent, so the parent will not die.
 */
unsafe fn weaken_task(f: fn(comm::Port<()>)) {
    let po = comm::Port();
    let ch = comm::Chan(po);
    unsafe {
        rustrt::rust_task_weaken(unsafe::reinterpret_cast(&ch));
    }
    let _unweaken = Unweaken(ch);
    f(po);

    struct Unweaken {
      let ch: comm::Chan<()>;
      new(ch: comm::Chan<()>) { self.ch = ch; }
      drop unsafe {
        rustrt::rust_task_unweaken(unsafe::reinterpret_cast(&self.ch));
      }
    }
}

#[test]
fn test_weaken_task_then_unweaken() {
    do task::try {
        unsafe {
            do weaken_task |_po| {
            }
        }
    };
}

#[test]
fn test_weaken_task_wait() {
    do task::spawn_unlinked {
        unsafe {
            do weaken_task |po| {
                comm::recv(po);
            }
        }
    }
}

#[test]
fn test_weaken_task_stress() {
    // Create a bunch of weak tasks
    for iter::repeat(100u) {
        do task::spawn {
            unsafe {
                do weaken_task |_po| {
                }
            }
        }
        do task::spawn_unlinked {
            unsafe {
                do weaken_task |po| {
                    // Wait for it to tell us to die
                    comm::recv(po);
                }
            }
        }
    }
}

#[test]
#[ignore(cfg(windows))]
fn test_weaken_task_fail() {
    let res = do task::try {
        unsafe {
            do weaken_task |_po| {
                fail;
            }
        }
    };
    assert result::is_err(res);
}
