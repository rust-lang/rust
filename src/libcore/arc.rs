/**
 * An atomically reference counted wrapper that can be used to
 * share immutable data between tasks.
 */

export arc, get, clone;

export exclusive, methods;

#[abi = "cdecl"]
extern mod rustrt {
    #[rust_stack]
    fn rust_atomic_increment(p: &mut libc::intptr_t)
        -> libc::intptr_t;

    #[rust_stack]
    fn rust_atomic_decrement(p: &mut libc::intptr_t)
        -> libc::intptr_t;
}

type arc_data<T> = {
    mut count: libc::intptr_t,
    data: T
};

class arc_destruct<T> {
  let data: *libc::c_void;
  new(data: *libc::c_void) { self.data = data; }
  drop unsafe {
     let data: ~arc_data<T> = unsafe::reinterpret_cast(self.data);
     let new_count = rustrt::rust_atomic_decrement(&mut data.count);
     assert new_count >= 0;
     if new_count == 0 {
         // drop glue takes over.
     } else {
       unsafe::forget(data);
     }
  }
}

type arc<T: const send> = arc_destruct<T>;

/// Create an atomically reference counted wrapper.
fn arc<T: const send>(-data: T) -> arc<T> {
    let data = ~{mut count: 1, data: data};
    unsafe {
        let ptr = unsafe::transmute(data);
        arc_destruct(ptr)
    }
}

/**
 * Access the underlying data in an atomically reference counted
 * wrapper.
 */
fn get<T: const send>(rc: &arc<T>) -> &T {
    unsafe {
        let ptr: ~arc_data<T> = unsafe::reinterpret_cast((*rc).data);
        // Cast us back into the correct region
        let r = unsafe::reinterpret_cast(&ptr.data);
        unsafe::forget(ptr);
        return r;
    }
}

/**
 * Duplicate an atomically reference counted wrapper.
 *
 * The resulting two `arc` objects will point to the same underlying data
 * object. However, one of the `arc` objects can be sent to another task,
 * allowing them to share the underlying data.
 */
fn clone<T: const send>(rc: &arc<T>) -> arc<T> {
    unsafe {
        let ptr: ~arc_data<T> = unsafe::reinterpret_cast((*rc).data);
        let new_count = rustrt::rust_atomic_increment(&mut ptr.count);
        assert new_count >= 2;
        unsafe::forget(ptr);
    }
    arc_destruct((*rc).data)
}

// An arc over mutable data that is protected by a lock.
type ex_data<T: send> =
    {lock: sys::little_lock, mut failed: bool, mut data: T};
type exclusive<T: send> = arc_destruct<ex_data<T>>;

fn exclusive<T:send >(-data: T) -> exclusive<T> {
    let data = ~{mut count: 1, data: {lock: sys::little_lock(), failed: false,
                                      data: data}};
    unsafe {
        let ptr = unsafe::reinterpret_cast(data);
        unsafe::forget(data);
        arc_destruct(ptr)
    }
}

impl<T: send> exclusive<T> {
    /// Duplicate an exclusive ARC. See arc::clone.
    fn clone() -> exclusive<T> {
        unsafe {
            // this makes me nervous...
            let ptr: ~arc_data<ex_data<T>> =
                  unsafe::reinterpret_cast(self.data);
            let new_count = rustrt::rust_atomic_increment(&mut ptr.count);
            assert new_count > 1;
            unsafe::forget(ptr);
        }
        arc_destruct(self.data)
    }

    /**
     * Access the underlying mutable data with mutual exclusion from other
     * tasks. The argument closure will be run with the mutex locked; all
     * other tasks wishing to access the data will block until the closure
     * finishes running.
     *
     * Currently, scheduling operations (i.e., yielding, receiving on a pipe,
     * accessing the provided condition variable) are prohibited while inside
     * the exclusive. Supporting that is a work in progress.
     *
     * The reason this function is 'unsafe' is because it is possible to
     * construct a circular reference among multiple ARCs by mutating the
     * underlying data. This creates potential for deadlock, but worse, this
     * will guarantee a memory leak of all involved ARCs. Using exclusive
     * ARCs inside of other ARCs is safe in absence of circular references.
     */
    unsafe fn with<U>(f: fn(x: &mut T) -> U) -> U {
        let ptr: ~arc_data<ex_data<T>> =
            unsafe::reinterpret_cast(self.data);
        assert ptr.count > 0;
        let ptr2: &arc_data<ex_data<T>> = unsafe::reinterpret_cast(&*ptr);
        unsafe::forget(ptr);
        let rec: &ex_data<T> = &(*ptr2).data;
        do rec.lock.lock {
            if rec.failed {
                fail ~"Poisoned arc::exclusive - another task failed inside!";
            }
            rec.failed = true;
            let result = f(&mut rec.data);
            rec.failed = false;
            result
        }
    }
}

#[cfg(test)]
mod tests {
    import comm::*;

    #[test]
    fn manually_share_arc() {
        let v = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let arc_v = arc::arc(v);

        let p = port();
        let c = chan(p);

        do task::spawn() {
            let p = port();
            c.send(chan(p));

            let arc_v = p.recv();

            let v = *arc::get::<~[int]>(&arc_v);
            assert v[3] == 4;
        };

        let c = p.recv();
        c.send(arc::clone(&arc_v));

        assert (*arc::get(&arc_v))[2] == 3;

        log(info, arc_v);
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
        let x = arc::exclusive(1);
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
