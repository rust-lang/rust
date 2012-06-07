#[doc = "An atomically reference counted wrapper that can be used to
share immutable data between tasks."]

import comm::{port, chan, methods};
import sys::methods;

export arc, get, clone, shared_arc, get_arc;

export exclusive, methods;

#[abi = "cdecl"]
native mod rustrt {
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

resource arc_destruct<T>(data: *libc::c_void) {
    unsafe {
        let data: ~arc_data<T> = unsafe::reinterpret_cast(data);
        let new_count = rustrt::rust_atomic_decrement(&mut data.count);
        let data_ptr : *() = unsafe::reinterpret_cast(data);
        assert new_count >= 0;
        if new_count == 0 {
            // drop glue takes over.
        } else {
            unsafe::forget(data);
        }
    }
}

type arc<T: const> = arc_destruct<T>;

#[doc="Create an atomically reference counted wrapper."]
fn arc<T: const>(-data: T) -> arc<T> {
    let data = ~{mut count: 1, data: data};
    unsafe {
        let ptr = unsafe::transmute(data);
        arc_destruct(ptr)
    }
}

#[doc="Access the underlying data in an atomically reference counted
 wrapper."]
fn get<T: const>(rc: &a.arc<T>) -> &a.T {
    unsafe {
        let ptr: ~arc_data<T> = unsafe::reinterpret_cast(**rc);
        // Cast us back into the correct region
        let r = unsafe::reinterpret_cast(&ptr.data);
        unsafe::forget(ptr);
        ret r;
    }
}

#[doc="Duplicate an atomically reference counted wrapper.

The resulting two `arc` objects will point to the same underlying data
object. However, one of the `arc` objects can be sent to another task,
allowing them to share the underlying data."]
fn clone<T: const>(rc: &arc<T>) -> arc<T> {
    unsafe {
        let ptr: ~arc_data<T> = unsafe::reinterpret_cast(**rc);
        let new_count = rustrt::rust_atomic_increment(&mut ptr.count);
        let data_ptr : *() = unsafe::reinterpret_cast(ptr);
        assert new_count >= 2;
        unsafe::forget(ptr);
    }
    arc_destruct(**rc)
}

// An arc over mutable data that is protected by a lock.
type ex_data<T: send> = {lock: sys::lock_and_signal, data: T};
type exclusive<T: send> = arc_destruct<ex_data<T>>;

fn exclusive<T:send >(-data: T) -> exclusive<T> {
    let data = ~{mut count: 1, data: {lock: sys::create_lock(),
                                      data: data}};
    unsafe {
        let ptr = unsafe::reinterpret_cast(data);
        unsafe::forget(data);
        arc_destruct(ptr)
    }
}

impl methods<T: send> for exclusive<T> {
    fn clone() -> exclusive<T> {
        unsafe {
            // this makes me nervous...
            let ptr: ~arc_data<ex_data<T>> = unsafe::reinterpret_cast(*self);
            let new_count = rustrt::rust_atomic_increment(&mut ptr.count);
            let data_ptr : *() = unsafe::reinterpret_cast(ptr);
            assert new_count > 1;
            unsafe::forget(ptr);
        }
        arc_destruct(*self)
    }

    fn with<U>(f: fn(sys::condition, x: &T) -> U) -> U {
        unsafe {
            let ptr: ~arc_data<ex_data<T>> = unsafe::reinterpret_cast(*self);
            let r = {
                let rec: &ex_data<T> = &(*ptr).data;
                rec.lock.lock_cond() {|c|
                    f(c, &rec.data)
                }
            };
            unsafe::forget(ptr);
            r
        }
    }
}

// Convenience code for sharing arcs between tasks

type get_chan<T: const send> = chan<chan<arc<T>>>;

// (terminate, get)
type shared_arc<T: const send> = (shared_arc_res, get_chan<T>);

resource shared_arc_res(c: comm::chan<()>) {
    c.send(());
}

fn shared_arc<T: send const>(-data: T) -> shared_arc<T> {
    let a = arc::arc(data);
    let p = port();
    let c = chan(p);
    task::spawn() {|move a|
        let mut live = true;
        let terminate = port();
        let get = port();

        c.send((chan(terminate), chan(get)));

        while live {
            alt comm::select2(terminate, get) {
              either::left(()) { live = false; }
              either::right(cc) {
                comm::send(cc, arc::clone(&a));
              }
            }
        }
    };
    let (terminate, get) = p.recv();
    (shared_arc_res(terminate), get)
}

fn get_arc<T: send const>(c: get_chan<T>) -> arc::arc<T> {
    let p = port();
    c.send(chan(p));
    p.recv()
}

#[cfg(test)]
mod tests {
    import comm::*;
    import future::future;

    #[test]
    fn manually_share_arc() {
        let v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let arc_v = arc::arc(v);

        let p = port();
        let c = chan(p);

        task::spawn() {||
            let p = port();
            c.send(chan(p));

            let arc_v = p.recv();

            let v = *arc::get::<[int]>(&arc_v);
            assert v[3] == 4;
        };

        let c = p.recv();
        c.send(arc::clone(&arc_v));

        assert (*arc::get(&arc_v))[2] == 3;

        log(info, arc_v);
    }

    #[test]
    fn auto_share_arc() {
        let v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let (_res, arc_c) = shared_arc(v);

        let p = port();
        let c = chan(p);

        task::spawn() {||
            let arc_v = get_arc(arc_c);
            let v = *get(&arc_v);
            assert v[2] == 3;

            c.send(());
        };

        assert p.recv() == ();
    }

    #[test]
    #[ignore] // this can probably infinite loop too.
    fn exclusive_arc() {
        let mut futures = [];

        let num_tasks = 10u;
        let count = 1000u;

        let total = exclusive(~mut 0u);

        for uint::range(0u, num_tasks) {|_i|
            let total = total.clone();
            futures += [future::spawn({||
                for uint::range(0u, count) {|_i|
                    total.with {|_cond, count|
                        **count += 1u;
                    }
                }
            })];
        };

        for futures.each {|f| f.get() };

        total.with {|_cond, total|
            assert **total == num_tasks * count
        };
    }
}
