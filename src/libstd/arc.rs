#[doc = "An atomically reference counted wrapper that can be used to
share immutable data between tasks."]

import comm::{port, chan, methods};

export arc, get, clone, shared_arc, get_arc;

#[abi = "cdecl"]
native mod rustrt {
    #[rust_stack]
    fn rust_atomic_increment(p: &mut libc::intptr_t)
        -> libc::intptr_t;

    #[rust_stack]
    fn rust_atomic_decrement(p: &mut libc::intptr_t)
        -> libc::intptr_t;
}

type arc_data<T: const> = {
    mut count: libc::intptr_t,
    data: T
};

resource arc_destruct<T: const>(data: *libc::c_void) {
    unsafe {
        let data: ~arc_data<T> = unsafe::reinterpret_cast(data);
        let ref_ptr = &mut data.count;

        let new_count = rustrt::rust_atomic_decrement(ref_ptr);
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
        let ptr = unsafe::reinterpret_cast(data);
        unsafe::forget(data);
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
        rustrt::rust_atomic_increment(&mut ptr.count);
        unsafe::forget(ptr);
    }
    arc_destruct(**rc)
}

// Convenience code for sharing arcs between tasks

enum proto<T: send const> {
    terminate,
    shared_get(comm::chan<arc::arc<T>>)
}

resource shared_arc_res<T: send const>(c: comm::chan<proto<T>>) {
    c.send(terminate);
}

fn shared_arc<T: send const>(-data: T) -> shared_arc_res<T> {
    let a = arc::arc(data);
    let c = task::spawn_listener::<proto<T>>() {|p, move a|
        let mut live = true;
        while live {
            alt p.recv() {
              terminate { live = false; }
              shared_get(cc) {
                cc.send(arc::clone(&a));
              }
            }
        }
    };
    shared_arc_res(c)
}

fn get_arc<T: send const>(c: comm::chan<proto<T>>) -> arc::arc<T> {
    let p = port();
    c.send(shared_get(chan(p)));
    p.recv()
}

#[cfg(test)]
mod tests {
    import comm::*;

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
}
