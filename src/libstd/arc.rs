#[doc = "An atomically reference counted wrapper that can be used to
share immutable data between tasks."]

export arc, get, clone;

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

type arc<T> = arc_destruct<T>;

#[doc="Create an atomically reference counted wrapper."]
fn arc<T>(-data: T) -> arc<T> {
    let data = ~{mut count: 1, data: data};
    unsafe {
        let ptr = unsafe::reinterpret_cast(data);
        unsafe::forget(data);
        arc_destruct(ptr)
    }
}

#[doc="Access the underlying data in an atomically reference counted
 wrapper."]
fn get<T>(rc: &a.arc<T>) -> &a.T {
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
fn clone<T>(rc: &arc<T>) -> arc<T> {
    unsafe {
        let ptr: ~arc_data<T> = unsafe::reinterpret_cast(**rc);
        rustrt::rust_atomic_increment(&mut ptr.count);
        unsafe::forget(ptr);
    }
    arc_destruct(**rc)
}
