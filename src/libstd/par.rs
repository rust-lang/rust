import future_spawn = future::spawn;

export map, mapi, alli, any, mapi_factory;

/**
 * The maximum number of tasks this module will spawn for a single
 * operation.
 */
const max_tasks : uint = 32u;

/// The minimum number of elements each task will process.
const min_granularity : uint = 1024u;

/**
 * An internal helper to map a function over a large vector and
 * return the intermediate results.
 *
 * This is used to build most of the other parallel vector functions,
 * like map or alli.
 */
fn map_slices<A: copy send, B: copy send>(
    xs: ~[A],
    f: fn() -> fn~(uint, v: &[A]) -> B)
    -> ~[B] {

    let len = xs.len();
    if len < min_granularity {
        log(info, ~"small slice");
        // This is a small vector, fall back on the normal map.
        ~[f()(0u, xs)]
    }
    else {
        let num_tasks = uint::min(max_tasks, len / min_granularity);

        let items_per_task = len / num_tasks;

        let mut futures = ~[];
        let mut base = 0u;
        log(info, ~"spawning tasks");
        while base < len {
            let end = uint::min(len, base + items_per_task);
            do vec::as_buf(xs) |p, _len| {
                let f = f();
                let f = do future_spawn() |copy base| {
                    unsafe {
                        let len = end - base;
                        let slice = (ptr::offset(p, base),
                                     len * sys::size_of::<A>());
                        log(info, fmt!("pre-slice: %?", (base, slice)));
                        let slice : &[A] =
                            unsafe::reinterpret_cast(slice);
                        log(info, fmt!("slice: %?",
                                       (base, vec::len(slice), end - base)));
                        assert(vec::len(slice) == end - base);
                        f(base, slice)
                    }
                };
                vec::push(futures, f);
            };
            base += items_per_task;
        }
        log(info, ~"tasks spawned");

        log(info, fmt!("num_tasks: %?", (num_tasks, futures.len())));
        assert(num_tasks == futures.len());

        let r = do futures.map() |ys| {
            ys.get()
        };
        assert(r.len() == futures.len());
        r
    }
}

/// A parallel version of map.
fn map<A: copy send, B: copy send>(xs: ~[A], f: fn~(A) -> B) -> ~[B] {
    vec::concat(map_slices(xs, || {
        fn~(_base: uint, slice : &[A], copy f) -> ~[B] {
            vec::map(slice, |x| f(x))
        }
    }))
}

/// A parallel version of mapi.
fn mapi<A: copy send, B: copy send>(xs: ~[A],
                                    f: fn~(uint, A) -> B) -> ~[B] {
    let slices = map_slices(xs, || {
        fn~(base: uint, slice : &[A], copy f) -> ~[B] {
            vec::mapi(slice, |i, x| {
                f(i + base, x)
            })
        }
    });
    let r = vec::concat(slices);
    log(info, (r.len(), xs.len()));
    assert(r.len() == xs.len());
    r
}

/**
 * A parallel version of mapi.
 *
 * In this case, f is a function that creates functions to run over the
 * inner elements. This is to skirt the need for copy constructors.
 */
fn mapi_factory<A: copy send, B: copy send>(
    xs: ~[A], f: fn() -> fn~(uint, A) -> B) -> ~[B] {
    let slices = map_slices(xs, || {
        let f = f();
        fn~(base: uint, slice : &[A], move f) -> ~[B] {
            vec::mapi(slice, |i, x| {
                f(i + base, x)
            })
        }
    });
    let r = vec::concat(slices);
    log(info, (r.len(), xs.len()));
    assert(r.len() == xs.len());
    r
}

/// Returns true if the function holds for all elements in the vector.
fn alli<A: copy send>(xs: ~[A], f: fn~(uint, A) -> bool) -> bool {
    do vec::all(map_slices(xs, || {
        fn~(base: uint, slice : &[A], copy f) -> bool {
            vec::alli(slice, |i, x| {
                f(i + base, x)
            })
        }
    })) |x| { x }
}

/// Returns true if the function holds for any elements in the vector.
fn any<A: copy send>(xs: ~[A], f: fn~(A) -> bool) -> bool {
    do vec::any(map_slices(xs, || {
        fn~(_base : uint, slice: &[A], copy f) -> bool {
            vec::any(slice, |x| f(x))
        }
    })) |x| { x }
}
