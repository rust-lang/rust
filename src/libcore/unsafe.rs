//! Unsafe operations

export reinterpret_cast, forget, bump_box_refcount, transmute;
export transmute_mut, transmute_immut, transmute_region, transmute_mut_region;

export SharedMutableState, shared_mutable_state, clone_shared_mutable_state;
export get_shared_mutable_state, get_shared_immutable_state;
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
unsafe fn transmute_mut<T>(+ptr: &T) -> &mut T { transmute(ptr) }
/// Coerce a mutable reference to be immutable.
unsafe fn transmute_immut<T>(+ptr: &mut T) -> &T { transmute(ptr) }
/// Coerce a borrowed pointer to have an arbitrary associated region.
unsafe fn transmute_region<T>(+ptr: &a/T) -> &b/T { transmute(ptr) }
/// Coerce a borrowed mutable pointer to have an arbitrary associated region.
unsafe fn transmute_mut_region<T>(+ptr: &a/mut T) -> &b/mut T {
    transmute(ptr)
}

/****************************************************************************
 * Shared state & exclusive ARC
 ****************************************************************************/

type ArcData<T> = {
    mut count: libc::intptr_t,
    data: T
};

class ArcDestruct<T> {
   let data: *libc::c_void;
   new(data: *libc::c_void) { self.data = data; }
   drop unsafe {
      let data: ~ArcData<T> = unsafe::reinterpret_cast(self.data);
      let new_count = rustrt::rust_atomic_decrement(&mut data.count);
      assert new_count >= 0;
      if new_count == 0 {
          // drop glue takes over.
      } else {
        unsafe::forget(data);
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
    let data = ~{mut count: 1, data: data};
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
        let r = unsafe::reinterpret_cast(&ptr.data);
        unsafe::forget(ptr);
        return r;
    }
}
#[inline(always)]
unsafe fn get_shared_immutable_state<T: send>(rc: &SharedMutableState<T>)
        -> &T {
    unsafe {
        let ptr: ~ArcData<T> = unsafe::reinterpret_cast((*rc).data);
        assert ptr.count > 0;
        // Cast us back into the correct region
        let r = unsafe::reinterpret_cast(&ptr.data);
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

    fn rust_create_little_lock() -> rust_little_lock;
    fn rust_destroy_little_lock(lock: rust_little_lock);
    fn rust_lock_little_lock(lock: rust_little_lock);
    fn rust_unlock_little_lock(lock: rust_little_lock);
}

class LittleLock {
    let l: rust_little_lock;
    new() {
        self.l = rustrt::rust_create_little_lock();
    }
    drop { rustrt::rust_destroy_little_lock(self.l); }
}

impl LittleLock {
    #[inline(always)]
    unsafe fn lock<T>(f: fn() -> T) -> T {
        class Unlock {
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
}
