// NB: transitionary, de-mode-ing.
// tjc: Re-forbid deprecated modes once a snapshot fixes the
// function problem
#[forbid(deprecated_pattern)];

#[doc(hidden)];

use compare_and_swap = rustrt::rust_compare_and_swap_ptr;
use task::TaskBuilder;
use task::atomically;

extern mod rustrt {
    #[legacy_exports];
    fn rust_task_weaken(ch: rust_port_id);
    fn rust_task_unweaken(ch: rust_port_id);

    #[rust_stack]
    fn rust_atomic_increment(p: &mut libc::intptr_t)
        -> libc::intptr_t;

    #[rust_stack]
    fn rust_atomic_decrement(p: &mut libc::intptr_t)
        -> libc::intptr_t;

    #[rust_stack]
    fn rust_compare_and_swap_ptr(address: &mut libc::uintptr_t,
                                 oldval: libc::uintptr_t,
                                 newval: libc::uintptr_t) -> bool;

    fn rust_create_little_lock() -> rust_little_lock;
    fn rust_destroy_little_lock(lock: rust_little_lock);
    fn rust_lock_little_lock(lock: rust_little_lock);
    fn rust_unlock_little_lock(lock: rust_little_lock);
}

#[allow(non_camel_case_types)] // runtime type
type rust_port_id = uint;

type GlobalPtr = *libc::uintptr_t;

/**
 * Atomically gets a channel from a pointer to a pointer-sized memory location
 * or, if no channel exists creates and installs a new channel and sets up a
 * new task to receive from it.
 */
pub unsafe fn chan_from_global_ptr<T: Send>(
    global: GlobalPtr,
    task_fn: fn() -> task::TaskBuilder,
    f: fn~(comm::Port<T>)
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
            |move f, setup_po, setup_ch| {
            let po = comm::Port::<T>();
            let ch = comm::Chan(&po);
            comm::send(setup_ch, ch);

            // Wait to hear if we are the official instance of
            // this global task
            match comm::recv::<Msg>(setup_po) {
              Proceed => f(move po),
              Abort => ()
            }
        };

        log(debug,~"before setup recv..");
        // This is the proposed global channel
        let ch = comm::recv(setup_po);
        // 0 is our sentinal value. It is not a valid channel
        assert *ch != 0;

        // Install the channel
        log(debug,~"BEFORE COMPARE AND SWAP");
        let swapped = compare_and_swap(
            cast::reinterpret_cast(&global),
            0u, cast::reinterpret_cast(&ch));
        log(debug,fmt!("AFTER .. swapped? %?", swapped));

        if swapped {
            // Success!
            comm::send(setup_ch, Proceed);
            ch
        } else {
            // Somebody else got in before we did
            comm::send(setup_ch, Abort);
            cast::reinterpret_cast(&*global)
        }
    } else {
        log(debug, ~"global != 0");
        cast::reinterpret_cast(&*global)
    }
}

#[test]
pub fn test_from_global_chan1() {

    // This is unreadable, right?

    // The global channel
    let globchan = 0;
    let globchanp = ptr::addr_of(&globchan);

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
    comm::send(ch, comm::Chan(&po));
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
    comm::send(ch, comm::Chan(&po));
    assert comm::recv(po) == true;
}

#[test]
pub fn test_from_global_chan2() {

    for iter::repeat(100) {
        // The global channel
        let globchan = 0;
        let globchanp = ptr::addr_of(&globchan);

        let resultpo = comm::Port();
        let resultch = comm::Chan(&resultpo);

        // Spawn a bunch of tasks that all want to compete to
        // create the global channel
        for uint::range(0, 10) |i| {
            do task::spawn {
                let ch = unsafe {
                    do chan_from_global_ptr(
                        globchanp, task::task) |po| {

                        for uint::range(0, 10) |_j| {
                            let ch = comm::recv(po);
                            comm::send(ch, {i});
                        }
                    }
                };
                let po = comm::Port();
                comm::send(ch, comm::Chan(&po));
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
pub unsafe fn weaken_task(f: fn(comm::Port<()>)) {
    let po = comm::Port();
    let ch = comm::Chan(&po);
    unsafe {
        rustrt::rust_task_weaken(cast::reinterpret_cast(&ch));
    }
    let _unweaken = Unweaken(ch);
    f(po);

    struct Unweaken {
      ch: comm::Chan<()>,
      drop unsafe {
        rustrt::rust_task_unweaken(cast::reinterpret_cast(&self.ch));
      }
    }

    fn Unweaken(ch: comm::Chan<()>) -> Unweaken {
        Unweaken {
            ch: ch
        }
    }
}

#[test]
pub fn test_weaken_task_then_unweaken() {
    do task::try {
        unsafe {
            do weaken_task |_po| {
            }
        }
    };
}

#[test]
pub fn test_weaken_task_wait() {
    do task::spawn_unlinked {
        unsafe {
            do weaken_task |po| {
                comm::recv(po);
            }
        }
    }
}

#[test]
pub fn test_weaken_task_stress() {
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
pub fn test_weaken_task_fail() {
    let res = do task::try {
        unsafe {
            do weaken_task |_po| {
                fail;
            }
        }
    };
    assert result::is_err(&res);
}

/****************************************************************************
 * Shared state & exclusive ARC
 ****************************************************************************/

// An unwrapper uses this protocol to communicate with the "other" task that
// drops the last refcount on an arc. Unfortunately this can't be a proper
// pipe protocol because the unwrapper has to access both stages at once.
type UnwrapProto = ~mut Option<(pipes::ChanOne<()>, pipes::PortOne<bool>)>;

struct ArcData<T> {
    mut count:     libc::intptr_t,
    mut unwrapper: libc::uintptr_t, // either a UnwrapProto or 0
    // FIXME(#3224) should be able to make this non-option to save memory, and
    // in unwrap() use "let ~ArcData { data: result, _ } = thing" to unwrap it
    mut data:      Option<T>,
}

struct ArcDestruct<T> {
    mut data: *libc::c_void,
    drop unsafe {
        if self.data.is_null() {
            return; // Happens when destructing an unwrapper's handle.
        }
        do task::unkillable {
            let data: ~ArcData<T> = cast::reinterpret_cast(&self.data);
            let new_count = rustrt::rust_atomic_decrement(&mut data.count);
            assert new_count >= 0;
            if new_count == 0 {
                // Were we really last, or should we hand off to an unwrapper?
                // It's safe to not xchg because the unwrapper will set the
                // unwrap lock *before* dropping his/her reference. In effect,
                // being here means we're the only *awake* task with the data.
                if data.unwrapper != 0 {
                    let p: UnwrapProto =
                        cast::reinterpret_cast(&data.unwrapper);
                    let (message, response) = option::swap_unwrap(p);
                    // Send 'ready' and wait for a response.
                    pipes::send_one(move message, ());
                    // Unkillable wait. Message guaranteed to come.
                    if pipes::recv_one(move response) {
                        // Other task got the data.
                        cast::forget(move data);
                    } else {
                        // Other task was killed. drop glue takes over.
                    }
                } else {
                    // drop glue takes over.
                }
            } else {
                cast::forget(move data);
            }
        }
    }
}

fn ArcDestruct<T>(data: *libc::c_void) -> ArcDestruct<T> {
    ArcDestruct {
        data: data
    }
}

pub unsafe fn unwrap_shared_mutable_state<T: Send>(rc: SharedMutableState<T>)
        -> T {
    struct DeathThroes<T> {
        mut ptr:      Option<~ArcData<T>>,
        mut response: Option<pipes::ChanOne<bool>>,
        drop unsafe {
            let response = option::swap_unwrap(&mut self.response);
            // In case we get killed early, we need to tell the person who
            // tried to wake us whether they should hand-off the data to us.
            if task::failing() {
                pipes::send_one(move response, false);
                // Either this swap_unwrap or the one below (at "Got here")
                // ought to run.
                cast::forget(option::swap_unwrap(&mut self.ptr));
            } else {
                assert self.ptr.is_none();
                pipes::send_one(move response, true);
            }
        }
    }

    do task::unkillable {
        let ptr: ~ArcData<T> = cast::reinterpret_cast(&rc.data);
        let (c1,p1) = pipes::oneshot(); // ()
        let (c2,p2) = pipes::oneshot(); // bool
        let server: UnwrapProto = ~mut Some((move c1,move p2));
        let serverp: libc::uintptr_t = cast::transmute(move server);
        // Try to put our server end in the unwrapper slot.
        if rustrt::rust_compare_and_swap_ptr(&mut ptr.unwrapper, 0, serverp) {
            // Got in. Step 0: Tell destructor not to run. We are now it.
            rc.data = ptr::null();
            // Step 1 - drop our own reference.
            let new_count = rustrt::rust_atomic_decrement(&mut ptr.count);
        //    assert new_count >= 0;
            if new_count == 0 {
                // We were the last owner. Can unwrap immediately.
                // Also we have to free the server endpoints.
                let _server: UnwrapProto = cast::transmute(move serverp);
                option::swap_unwrap(&mut ptr.data)
                // drop glue takes over.
            } else {
                // The *next* person who sees the refcount hit 0 will wake us.
                let end_result =
                    DeathThroes { ptr: Some(move ptr),
                                  response: Some(move c2) };
                let mut p1 = Some(move p1); // argh
                do task::rekillable {
                    pipes::recv_one(option::swap_unwrap(&mut p1));
                }
                // Got here. Back in the 'unkillable' without getting killed.
                // Recover ownership of ptr, then take the data out.
                let ptr = option::swap_unwrap(&mut end_result.ptr);
                option::swap_unwrap(&mut ptr.data)
                // drop glue takes over.
            }
        } else {
            // Somebody else was trying to unwrap. Avoid guaranteed deadlock.
            cast::forget(move ptr);
            // Also we have to free the (rejected) server endpoints.
            let _server: UnwrapProto = cast::transmute(move serverp);
            fail ~"Another task is already unwrapping this ARC!";
        }
    }
}

/**
 * COMPLETELY UNSAFE. Used as a primitive for the safe versions in std::arc.
 *
 * Data races between tasks can result in crashes and, with sufficient
 * cleverness, arbitrary type coercion.
 */
pub type SharedMutableState<T: Send> = ArcDestruct<T>;

pub unsafe fn shared_mutable_state<T: Send>(data: T) ->
        SharedMutableState<T> {
    let data = ~ArcData { count: 1, unwrapper: 0, data: Some(move data) };
    unsafe {
        let ptr = cast::transmute(move data);
        ArcDestruct(ptr)
    }
}

#[inline(always)]
pub unsafe fn get_shared_mutable_state<T: Send>(rc: &a/SharedMutableState<T>)
        -> &a/mut T {
    unsafe {
        let ptr: ~ArcData<T> = cast::reinterpret_cast(&(*rc).data);
        assert ptr.count > 0;
        // Cast us back into the correct region
        let r = cast::transmute_region(option::get_ref(&ptr.data));
        cast::forget(move ptr);
        return cast::transmute_mut(r);
    }
}
#[inline(always)]
pub unsafe fn get_shared_immutable_state<T: Send>(
        rc: &a/SharedMutableState<T>) -> &a/T {
    unsafe {
        let ptr: ~ArcData<T> = cast::reinterpret_cast(&(*rc).data);
        assert ptr.count > 0;
        // Cast us back into the correct region
        let r = cast::transmute_region(option::get_ref(&ptr.data));
        cast::forget(move ptr);
        return r;
    }
}

pub unsafe fn clone_shared_mutable_state<T: Send>(rc: &SharedMutableState<T>)
        -> SharedMutableState<T> {
    unsafe {
        let ptr: ~ArcData<T> = cast::reinterpret_cast(&(*rc).data);
        let new_count = rustrt::rust_atomic_increment(&mut ptr.count);
        assert new_count >= 2;
        cast::forget(move ptr);
    }
    ArcDestruct((*rc).data)
}

/****************************************************************************/

#[allow(non_camel_case_types)] // runtime type
type rust_little_lock = *libc::c_void;

struct LittleLock {
    l: rust_little_lock,
    drop { rustrt::rust_destroy_little_lock(self.l); }
}

fn LittleLock() -> LittleLock {
    LittleLock {
        l: rustrt::rust_create_little_lock()
    }
}

impl LittleLock {
    #[inline(always)]
    unsafe fn lock<T>(f: fn() -> T) -> T {
        struct Unlock {
            l: rust_little_lock,
            drop { rustrt::rust_unlock_little_lock(self.l); }
        }

        fn Unlock(l: rust_little_lock) -> Unlock {
            Unlock {
                l: l
            }
        }

        do atomically {
            rustrt::rust_lock_little_lock(self.l);
            let _r = Unlock(self.l);
            f()
        }
    }
}

struct ExData<T: Send> { lock: LittleLock, mut failed: bool, mut data: T, }
/**
 * An arc over mutable data that is protected by a lock. For library use only.
 */
pub struct Exclusive<T: Send> { x: SharedMutableState<ExData<T>> }

pub fn exclusive<T:Send >(user_data: T) -> Exclusive<T> {
    let data = ExData {
        lock: LittleLock(), mut failed: false, mut data: move user_data
    };
    Exclusive { x: unsafe { shared_mutable_state(move data) } }
}

impl<T: Send> Exclusive<T> {
    // Duplicate an exclusive ARC, as std::arc::clone.
    fn clone() -> Exclusive<T> {
        Exclusive { x: unsafe { clone_shared_mutable_state(&self.x) } }
    }

    // Exactly like std::arc::mutex_arc,access(), but with the little_lock
    // instead of a proper mutex. Same reason for being unsafe.
    //
    // Currently, scheduling operations (i.e., yielding, receiving on a pipe,
    // accessing the provided condition variable) are prohibited while inside
    // the exclusive. Supporting that is a work in progress.
    #[inline(always)]
    unsafe fn with<U>(f: fn(x: &mut T) -> U) -> U {
        let rec = unsafe { get_shared_mutable_state(&self.x) };
        do rec.lock.lock {
            if rec.failed {
                fail ~"Poisoned exclusive - another task failed inside!";
            }
            rec.failed = true;
            let result = f(&mut rec.data);
            rec.failed = false;
            move result
        }
    }

    #[inline(always)]
    unsafe fn with_imm<U>(f: fn(x: &T) -> U) -> U {
        do self.with |x| {
            f(cast::transmute_immut(x))
        }
    }
}

// FIXME(#3724) make this a by-move method on the exclusive
pub fn unwrap_exclusive<T: Send>(arc: Exclusive<T>) -> T {
    let Exclusive { x: x } <- arc;
    let inner = unsafe { unwrap_shared_mutable_state(move x) };
    let ExData { data: data, _ } <- inner;
    move data
}

#[cfg(test)]
pub mod tests {
    #[test]
    pub fn exclusive_arc() {
        let mut futures = ~[];

        let num_tasks = 10;
        let count = 10;

        let total = exclusive(~mut 0);

        for uint::range(0, num_tasks) |_i| {
            let total = total.clone();
            futures.push(future::spawn(|move total| {
                for uint::range(0, count) |_i| {
                    do total.with |count| {
                        **count += 1;
                    }
                }
            }));
        };

        for futures.each |f| { f.get() }

        do total.with |total| {
            assert **total == num_tasks * count
        };
    }

    #[test] #[should_fail] #[ignore(cfg(windows))]
    pub fn exclusive_poison() {
        // Tests that if one task fails inside of an exclusive, subsequent
        // accesses will also fail.
        let x = exclusive(1);
        let x2 = x.clone();
        do task::try |move x2| {
            do x2.with |one| {
                assert *one == 2;
            }
        };
        do x.with |one| {
            assert *one == 1;
        }
    }

    #[test]
    pub fn exclusive_unwrap_basic() {
        let x = exclusive(~~"hello");
        assert unwrap_exclusive(move x) == ~~"hello";
    }

    #[test]
    pub fn exclusive_unwrap_contended() {
        let x = exclusive(~~"hello");
        let x2 = ~mut Some(x.clone());
        do task::spawn |move x2| {
            let x2 = option::swap_unwrap(x2);
            do x2.with |_hello| { }
            task::yield();
        }
        assert unwrap_exclusive(move x) == ~~"hello";

        // Now try the same thing, but with the child task blocking.
        let x = exclusive(~~"hello");
        let x2 = ~mut Some(x.clone());
        let mut res = None;
        do task::task().future_result(|+r| res = Some(move r)).spawn
              |move x2| {
            let x2 = option::swap_unwrap(x2);
            assert unwrap_exclusive(move x2) == ~~"hello";
        }
        // Have to get rid of our reference before blocking.
        { let _x = move x; } // FIXME(#3161) util::ignore doesn't work here
        let res = option::swap_unwrap(&mut res);
        future::get(&res);
    }

    #[test] #[should_fail] #[ignore(cfg(windows))]
    pub fn exclusive_unwrap_conflict() {
        let x = exclusive(~~"hello");
        let x2 = ~mut Some(x.clone());
        let mut res = None;
        do task::task().future_result(|+r| res = Some(move r)).spawn
           |move x2| {
            let x2 = option::swap_unwrap(x2);
            assert unwrap_exclusive(move x2) == ~~"hello";
        }
        assert unwrap_exclusive(move x) == ~~"hello";
        let res = option::swap_unwrap(&mut res);
        future::get(&res);
    }

    #[test] #[ignore(cfg(windows))]
    pub fn exclusive_unwrap_deadlock() {
        // This is not guaranteed to get to the deadlock before being killed,
        // but it will show up sometimes, and if the deadlock were not there,
        // the test would nondeterministically fail.
        let result = do task::try {
            // a task that has two references to the same exclusive will
            // deadlock when it unwraps. nothing to be done about that.
            let x = exclusive(~~"hello");
            let x2 = x.clone();
            do task::spawn {
                for 10.times { task::yield(); } // try to let the unwrapper go
                fail; // punt it awake from its deadlock
            }
            let _z = unwrap_exclusive(move x);
            do x2.with |_hello| { }
        };
        assert result.is_err();
    }
}
