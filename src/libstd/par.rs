import comm::port;
import comm::chan;
import comm::send;
import comm::recv;
import task::spawn;

export future;
export map;
export alli;

iface future<T: send> {
    fn get() -> T;
}

type future_<T: send> = {
    mut slot : option<T>,
    port : port<T>,
};

impl<T: send> of future<T> for future_<T> {
    fn get() -> T {
        alt(self.slot) {
          some(x) { x }
          none {
            let x = recv(self.port);
            self.slot = some(x);
            x
          }
        }
    }
}


#[doc="Executes a bit of code asynchronously.

Returns a handle that can be used to retrieve the result at your
leisure."]
fn future<T: send>(thunk : fn~() -> T) -> future<T> {
    let p = port();
    let c = chan(p);

    spawn() {||
        send(c, thunk());
    }

    {mut slot: none::<T>, port : p} as future::<T>
}

#[doc="The maximum number of tasks this module will spawn for a single
 operationg."]
const max_tasks : uint = 32u;

#[doc="The minimum number of elements each task will process."]
const min_granularity : uint = 1024u;

#[doc="An internal helper to map a function over a large vector and
 return the intermediate results.

This is used to build most of the other parallel vector functions,
like map or alli."]
fn map_slices<A: send, B: send>(xs: [A], f: fn~(uint, [A]) -> B) -> [B] {
    let len = xs.len();
    if len < min_granularity {
        // This is a small vector, fall back on the normal map.
        [f(0u, xs)]
    }
    else {
        let num_tasks = uint::min(max_tasks, len / min_granularity);

        let items_per_task = len / num_tasks;

        let mut futures = [];
        let mut base = 0u;
        while base < len {
            let slice = vec::slice(xs, base,
                                   uint::min(len, base + items_per_task));
            futures += [future() {|copy base|
                f(base, slice)
            }];
            base += items_per_task;
        }

        futures.map() {|ys|
            ys.get()
        }
    }
}

#[doc="A parallel version of map."]
fn map<A: send, B: send>(xs: [A], f: fn~(A) -> B) -> [B] {
    vec::concat(map_slices(xs) {|_base, slice|
        map(slice, f)
    })
}

#[doc="Returns true if the function holds for all elements in the vector."]
fn alli<A: send>(xs: [A], f: fn~(uint, A) -> bool) -> bool {
    vec::all(map_slices(xs) {|base, slice|
        slice.alli() {|i, x|
            f(i + base, x)
        }
    }) {|x| x }
}
