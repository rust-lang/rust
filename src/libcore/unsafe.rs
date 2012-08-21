//! Unsafe operations

export reinterpret_cast, forget, bump_box_refcount, transmute;
export transmute_mut, transmute_immut, transmute_region, transmute_mut_region;

export SharedMutableState, shared_mutable_state, clone_shared_mutable_state;
export get_shared_mutable_state, get_shared_immutable_state;
export unwrap_shared_mutable_state;
export Exclusive, exclusive;

import task::atomically;

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn forget<T>(-x: T);
    fn reinterpret_cast<T, U>(e: T) -> U;
}

/// Casts the value at `src` to U. The two types must have the same length.
#[inline(always)]
unsafe fn reinterpret_cast<T, U>(src: T) -> U {
    rusti::reinterpret_cast(src)
}

/**
 * Move a thing into the void
 *
 * The forget function will take ownership of the provided value but neglect
 * to run any required cleanup or memory-management operations on it. This
 * can be used for various acts of magick, particularly when using
 * reinterpret_cast on managed pointer types.
 */
#[inline(always)]
unsafe fn forget<T>(-thing: T) { rusti::forget(thing); }

/**
 * Force-increment the reference count on a shared box. If used
 * uncarefully, this can leak the box. Use this in conjunction with transmute
 * and/or reinterpret_cast when such calls would otherwise scramble a box's
 * reference count
 */
unsafe fn bump_box_refcount<T>(+t: @T) { forget(t); }

/**
 * Transform a value of one type into a value of another type.
 * Both types must have the same size and alignment.
 *
 * # Example
 *
 *     assert transmute("L") == ~[76u8, 0u8];
 */
unsafe fn transmute<L, G>(-thing: L) -> G {
    let newthing = reinterpret_cast(thing);
    forget(thing);
    return newthing;
}

/// Coerce an immutable reference to be mutable.
unsafe fn transmute_mut<T>(+ptr: &a/T) -> &a/mut T { transmute(ptr) }
/// Coerce a mutable reference to be immutable.
unsafe fn transmute_immut<T>(+ptr: &a/mut T) -> &a/T { transmute(ptr) }
/// Coerce a borrowed pointer to have an arbitrary associated region.
unsafe fn transmute_region<T>(+ptr: &a/T) -> &b/T { transmute(ptr) }
/// Coerce a borrowed mutable pointer to have an arbitrary associated region.
unsafe fn transmute_mut_region<T>(+ptr: &a/mut T) -> &b/mut T {
    transmute(ptr)
}

/****************************************************************************
 * Shared state & exclusive ARC
 ****************************************************************************/

// An unwrapper uses this protocol to communicate with the "other" task that
// drops the last refcount on an arc. Unfortunately this can't be a proper
// pipe protocol because the unwrapper has to access both stages at once.
type UnwrapProto = ~mut option<(pipes::chan_one<()>, pipes::port_one<bool>)>;

struct ArcData<T> {
    mut count:     libc::intptr_t;
    mut unwrapper: libc::uintptr_t; // either a UnwrapProto or 0
    // FIXME(#3224) should be able to make this non-option to save memory, and
    // in unwrap() use "let ~ArcData { data: result, _ } = thing" to unwrap it
    mut data:      option<T>;
}

struct ArcDestruct<T> {
    mut data: *libc::c_void;
    new(data: *libc::c_void) { self.data = data; }
    drop unsafe {
        if self.data.is_null() {
            return; // Happens when destructing an unwrapper's handle.
        }
        do task::unkillable {
            let data: ~ArcData<T> = unsafe::reinterpret_cast(self.data);
            let new_count = rustrt::rust_atomic_decrement(&mut data.count);
            assert new_count >= 0;
            if new_count == 0 {
                // Were we really last, or should we hand off to an unwrapper?
                // It's safe to not xchg because the unwrapper will set the
                // unwrap lock *before* dropping his/her reference. In effect,
                // being here means we're the only *awake* task with the data.
                if data.unwrapper != 0 {
                    let p: UnwrapProto =
                        unsafe::reinterpret_cast(data.unwrapper);
                    let (message, response) = option::swap_unwrap(p);
                    // Send 'ready' and wait for a response.
                    pipes::send_one(message, ());
                    // Unkillable wait. Message guaranteed to come.
                    if pipes::recv_one(response) {
                        // Other task got the data.
                        unsafe::forget(data);
                    } else {
                        // Other task was killed. drop glue takes over.
                    }
                } else {
                    // drop glue takes over.
                }
            } else {
                unsafe::forget(data);
            }
        }
    }
}

unsafe fn unwrap_shared_mutable_state<T: send>(+rc: SharedMutableState<T>)
        -> T {
    struct DeathThroes<T> {
        mut ptr:      option<~ArcData<T>>;
        mut response: option<pipes::chan_one<bool>>;
        drop unsafe {
            let response = option::swap_unwrap(&mut self.response);
            // In case we get killed early, we need to tell the person who
            // tried to wake us whether they should hand-off the data to us.
            if task::failing() {
                pipes::send_one(response, false);
                // Either this swap_unwrap or the one below (at "Got here")
                // ought to run.
                unsafe::forget(option::swap_unwrap(&mut self.ptr));
            } else {
                assert self.ptr.is_none();
                pipes::send_one(response, true);
            }
        }
    }

    do task::unkillable {
        let ptr: ~ArcData<T> = unsafe::reinterpret_cast(rc.data);
        let (c1,p1) = pipes::oneshot(); // ()
        let (c2,p2) = pipes::oneshot(); // bool
        let server: UnwrapProto = ~mut some((c1,p2));
        let serverp: libc::uintptr_t = unsafe::transmute(server);
        // Try to put our server end in the unwrapper slot.
        if rustrt::rust_compare_and_swap_ptr(&mut ptr.unwrapper, 0, serverp) {
            // Got in. Step 0: Tell destructor not to run. We are now it.
            rc.data = ptr::null();
            // Step 1 - drop our own reference.
            let new_count = rustrt::rust_atomic_decrement(&mut ptr.count);
            assert new_count >= 0;
            if new_count == 0 {
                // We were the last owner. Can unwrap immediately.
                // Also we have to free the server endpoints.
                let _server: UnwrapProto = unsafe::transmute(serverp);
                option::swap_unwrap(&mut ptr.data)
                // drop glue takes over.
            } else {
                // The *next* person who sees the refcount hit 0 will wake us.
                let end_result =
                    DeathThroes { ptr: some(ptr), response: some(c2) };
                let mut p1 = some(p1); // argh
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
            unsafe::forget(ptr);
            // Also we have to free the (rejected) server endpoints.
            let _server: UnwrapProto = unsafe::transmute(serverp);
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
type SharedMutableState<T: send> = ArcDestruct<T>;

unsafe fn shared_mutable_state<T: send>(+data: T) -> SharedMutableState<T> {
    let data = ~ArcData { count: 1, unwrapper: 0, data: some(data) };
    unsafe {
        let ptr = unsafe::transmute(data);
        ArcDestruct(ptr)
    }
}

#[inline(always)]
unsafe fn get_shared_mutable_state<T: send>(rc: &SharedMutableState<T>)
        -> &mut T {
    unsafe {
        let ptr: ~ArcData<T> = unsafe::reinterpret_cast((*rc).data);
        assert ptr.count > 0;
        // Cast us back into the correct region
        let r = unsafe::transmute_region(option::get_ref(&ptr.data));
        unsafe::forget(ptr);
        return unsafe::transmute_mut(r);
    }
}
#[inline(always)]
unsafe fn get_shared_immutable_state<T: send>(rc: &SharedMutableState<T>)
        -> &T {
    unsafe {
        let ptr: ~ArcData<T> = unsafe::reinterpret_cast((*rc).data);
        assert ptr.count > 0;
        // Cast us back into the correct region
        let r = unsafe::transmute_region(option::get_ref(&ptr.data));
        unsafe::forget(ptr);
        return r;
    }
}

unsafe fn clone_shared_mutable_state<T: send>(rc: &SharedMutableState<T>)
        -> SharedMutableState<T> {
    unsafe {
        let ptr: ~ArcData<T> = unsafe::reinterpret_cast((*rc).data);
        let new_count = rustrt::rust_atomic_increment(&mut ptr.count);
        assert new_count >= 2;
        unsafe::forget(ptr);
    }
    ArcDestruct((*rc).data)
}

/****************************************************************************/

#[allow(non_camel_case_types)] // runtime type
type rust_little_lock = *libc::c_void;

#[abi = "cdecl"]
extern mod rustrt {
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

struct LittleLock {
    let l: rust_little_lock;
    new() {
        self.l = rustrt::rust_create_little_lock();
    }
    drop { rustrt::rust_destroy_little_lock(self.l); }
}

impl LittleLock {
    #[inline(always)]
    unsafe fn lock<T>(f: fn() -> T) -> T {
        struct Unlock {
            let l: rust_little_lock;
            new(l: rust_little_lock) { self.l = l; }
            drop { rustrt::rust_unlock_little_lock(self.l); }
        }

        do atomically {
            rustrt::rust_lock_little_lock(self.l);
            let _r = Unlock(self.l);
            f()
        }
    }
}

struct ExData<T: send> { lock: LittleLock; mut failed: bool; mut data: T; }
/**
 * An arc over mutable data that is protected by a lock. For library use only.
 */
struct Exclusive<T: send> { x: SharedMutableState<ExData<T>>; }

fn exclusive<T:send >(+user_data: T) -> Exclusive<T> {
    let data = ExData {
        lock: LittleLock(), mut failed: false, mut data: user_data
    };
    Exclusive { x: unsafe { shared_mutable_state(data) } }
}

impl<T: send> Exclusive<T> {
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
            result
        }
    }
}

// FIXME(#2585) make this a by-move method on the exclusive
fn unwrap_exclusive<T: send>(+arc: Exclusive<T>) -> T {
    let Exclusive { x: x } = arc;
    let inner = unsafe { unwrap_shared_mutable_state(x) };
    let ExData { data: data, _ } = inner;
    data
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {

    #[test]
    fn test_reinterpret_cast() {
        assert unsafe { reinterpret_cast(1) } == 1u;
    }

    #[test]
    fn test_bump_box_refcount() {
        unsafe {
            let box = @~"box box box";       // refcount 1
            bump_box_refcount(box);         // refcount 2
            let ptr: *int = transmute(box); // refcount 2
            let _box1: @~str = reinterpret_cast(ptr);
            let _box2: @~str = reinterpret_cast(ptr);
            assert *_box1 == ~"box box box";
            assert *_box2 == ~"box box box";
            // Will destroy _box1 and _box2. Without the bump, this would
            // use-after-free. With too many bumps, it would leak.
        }
    }

    #[test]
    fn test_transmute() {
        unsafe {
            let x = @1;
            let x: *int = transmute(x);
            assert *x == 1;
            let _x: @int = transmute(x);
        }
    }

    #[test]
    fn test_transmute2() {
        unsafe {
            assert transmute(~"L") == ~[76u8, 0u8];
        }
    }

    #[test]
    fn exclusive_arc() {
        let mut futures = ~[];

        let num_tasks = 10u;
        let count = 10u;

        let total = exclusive(~mut 0u);

        for uint::range(0u, num_tasks) |_i| {
            let total = total.clone();
            futures += ~[future::spawn(|| {
                for uint::range(0u, count) |_i| {
                    do total.with |count| {
                        **count += 1u;
                    }
                }
            })];
        };

        for futures.each |f| { f.get() }

        do total.with |total| {
            assert **total == num_tasks * count
        };
    }

    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn exclusive_poison() {
        // Tests that if one task fails inside of an exclusive, subsequent
        // accesses will also fail.
        let x = exclusive(1);
        let x2 = x.clone();
        do task::try {
            do x2.with |one| {
                assert *one == 2;
            }
        };
        do x.with |one| {
            assert *one == 1;
        }
    }

    #[test]
    fn exclusive_unwrap_basic() {
        let x = exclusive(~~"hello");
        assert unwrap_exclusive(x) == ~~"hello";
    }

    #[test]
    fn exclusive_unwrap_contended() {
        let x = exclusive(~~"hello");
        let x2 = ~mut some(x.clone());
        do task::spawn {
            let x2 = option::swap_unwrap(x2);
            do x2.with |_hello| { }
            task::yield();
        }
        assert unwrap_exclusive(x) == ~~"hello";

        // Now try the same thing, but with the child task blocking.
        let x = exclusive(~~"hello");
        let x2 = ~mut some(x.clone());
        let mut res = none;
        do task::task().future_result(|+r| res = some(r)).spawn {
            let x2 = option::swap_unwrap(x2);
            assert unwrap_exclusive(x2) == ~~"hello";
        }
        // Have to get rid of our reference before blocking.
        { let _x = move x; } // FIXME(#3161) util::ignore doesn't work here
        let res = option::swap_unwrap(&mut res);
        future::get(&res);
    }

    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn exclusive_unwrap_conflict() {
        let x = exclusive(~~"hello");
        let x2 = ~mut some(x.clone());
        let mut res = none;
        do task::task().future_result(|+r| res = some(r)).spawn {
            let x2 = option::swap_unwrap(x2);
            assert unwrap_exclusive(x2) == ~~"hello";
        }
        assert unwrap_exclusive(x) == ~~"hello";
        let res = option::swap_unwrap(&mut res);
        future::get(&res);
    }

    #[test] #[ignore(cfg(windows))]
    fn exclusive_unwrap_deadlock() {
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
            let _z = unwrap_exclusive(x);
            do x2.with |_hello| { }
        };
        assert result.is_err();
    }
}
