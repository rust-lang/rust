#[doc(hidden)];

export chan_from_global_ptr;

import compare_and_swap = rustrt::rust_compare_and_swap_ptr;

native mod rustrt {
    fn rust_compare_and_swap_ptr(address: *libc::uintptr_t,
                                 oldval: libc::uintptr_t,
                                 newval: libc::uintptr_t) -> bool;
}

type global_ptr<T: send> = *libc::uintptr_t;

#[doc = "
Atomically gets a channel from a pointer to a pointer-sized memory location
or, if no channel exists creates and installs a new channel and sets up a new
task to receive from it.
"]
unsafe fn chan_from_global_ptr<T: send>(
    global: global_ptr<T>,
    builder: fn() -> task::builder,
    f: fn~(comm::port<T>)
) -> comm::chan<T> {

    enum msg {
        proceed,
        abort
    }

    let is_probably_zero = *global == 0u;
    if is_probably_zero {
        // There's no global channel. We must make it

        let setup_po = comm::port();
        let setup_ch = comm::chan(setup_po);
        let setup_ch = task::run_listener(builder()) {|setup_po|
            let po = comm::port::<T>();
            let ch = comm::chan(po);
            comm::send(setup_ch, ch);

            // Wait to hear if we are the official instance of
            // this global task
            alt comm::recv::<msg>(setup_po) {
              proceed { f(po); }
              abort { }
            }
        };

        // This is the proposed global channel
        let ch = comm::recv(setup_po);
        // 0 is our sentinal value. It is not a valid channel
        assert unsafe::reinterpret_cast(ch) != 0u;

        // Install the channel
        let swapped = compare_and_swap(
            global, 0u, unsafe::reinterpret_cast(ch));

        if swapped {
            // Success!
            comm::send(setup_ch, proceed);
            ch
        } else {
            // Somebody else got in before we did
            comm::send(setup_ch, abort);
            unsafe::reinterpret_cast(*global)
        }
    } else {
        unsafe::reinterpret_cast(*global)
    }
}

#[test]
fn test_from_global_chan1() unsafe {

    // This is unreadable, right?

    // The global channel
    let globchan = 0u;
    let globchanp = ptr::addr_of(globchan);

    // Create the global channel, attached to a new task
    let ch = chan_from_global_ptr(globchanp, task::builder) {|po|
        let ch = comm::recv(po);
        comm::send(ch, true);
        let ch = comm::recv(po);
        comm::send(ch, true);
    };
    // Talk to it
    let po = comm::port();
    comm::send(ch, comm::chan(po));
    assert comm::recv(po) == true;

    // This one just reuses the previous channel
    let ch = chan_from_global_ptr(globchanp, task::builder) {|po|
        let ch = comm::recv(po);
        comm::send(ch, false);
    };

    // Talk to the original global task
    let po = comm::port();
    comm::send(ch, comm::chan(po));
    assert comm::recv(po) == true;
}

#[test]
fn test_from_global_chan2() unsafe {

    iter::repeat(100u) {||
        // The global channel
        let globchan = 0u;
        let globchanp = ptr::addr_of(globchan);

        let resultpo = comm::port();
        let resultch = comm::chan(resultpo);

        // Spawn a bunch of tasks that all want to compete to
        // create the global channel
        uint::range(0u, 10u) {|i|
            task::spawn() {||
                let ch = chan_from_global_ptr(
                    globchanp, task::builder) {|po|

                    uint::range(0u, 10u) {|_j|
                        let ch = comm::recv(po);
                        comm::send(ch, {i});
                    }
                };
                let po = comm::port();
                comm::send(ch, comm::chan(po));
                // We are the winner if our version of the
                // task was installed
                let winner = comm::recv(po);
                comm::send(resultch, winner == i);
            }
        }
        // There should be only one winner
        let mut winners = 0u;
        uint::range(0u, 10u) {|_i|
            let res = comm::recv(resultpo);
            if res { winners += 1u };
        }
        assert winners == 1u;
    }
}
