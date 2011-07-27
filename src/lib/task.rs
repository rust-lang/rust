native "rust" mod rustrt {
    fn task_sleep(time_in_us: uint);
    fn task_yield();
    fn task_join(t: task) -> int;
    fn unsupervise();
    fn pin_task();
    fn unpin_task();
    fn clone_chan(c: *rust_chan) -> *rust_chan;

    type rust_chan;
}

/**
 * Hints the scheduler to yield this task for a specified ammount of time.
 *
 * arg: time_in_us maximum number of microseconds to yield control for
 */
fn sleep(time_in_us: uint) { ret rustrt::task_sleep(time_in_us); }

fn yield() { ret rustrt::task_yield(); }

tag task_result { tr_success; tr_failure; }

fn join(t: task) -> task_result {
    alt rustrt::task_join(t) { 0 { tr_success } _ { tr_failure } }
}

fn unsupervise() { ret rustrt::unsupervise(); }

fn pin() { rustrt::pin_task(); }

fn unpin() { rustrt::unpin_task(); }

fn clone_chan[T](c: chan[T]) -> chan[T] {
    let cloned = rustrt::clone_chan(unsafe::reinterpret_cast(c));
    ret unsafe::reinterpret_cast(cloned);
}

fn send[T](c: chan[T], v: &T) { c <| v; }

fn recv[T](p: port[T]) -> T { let v; p |> v; v }

// Spawn a task and immediately return a channel for communicating to it
fn worker[T](f: fn(port[T]) ) -> {task: task, chan: chan[T]} {
    // FIXME: This is frighteningly unsafe and only works for
    // a few cases

    type opaque = int;

    // FIXME: This terrible hackery is because worktask can't currently
    // have type params
    type wordsz1 = int;
    type wordsz2 = {a: int, b: int};
    type wordsz3 = {a: int, b: int, c: int};
    type wordsz4 = {a: int, b: int, c: int, d: int};
    type wordsz5 = {a: int, b: int, c: int, d: int, e: int};
    type opaquechan_1wordsz = chan[chan[wordsz1]];
    type opaquechan_2wordsz = chan[chan[wordsz2]];
    type opaquechan_3wordsz = chan[chan[wordsz3]];
    type opaquechan_4wordsz = chan[chan[wordsz4]];
    type opaquechan_5wordsz = chan[chan[wordsz5]];

    fn worktask1(setupch: opaquechan_1wordsz, fptr: opaque) {
        let f: *fn(port[wordsz1])  = unsafe::reinterpret_cast(fptr);
        let p = port[wordsz1]();
        setupch <| chan(p);
        (*f)(p);
    }

    fn worktask2(setupch: opaquechan_2wordsz, fptr: opaque) {
        let f: *fn(port[wordsz2])  = unsafe::reinterpret_cast(fptr);
        let p = port[wordsz2]();
        setupch <| chan(p);
        (*f)(p);
    }

    fn worktask3(setupch: opaquechan_3wordsz, fptr: opaque) {
        let f: *fn(port[wordsz3])  = unsafe::reinterpret_cast(fptr);
        let p = port[wordsz3]();
        setupch <| chan(p);
        (*f)(p);
    }

    fn worktask4(setupch: opaquechan_4wordsz, fptr: opaque) {
        let f: *fn(port[wordsz4])  = unsafe::reinterpret_cast(fptr);
        let p = port[wordsz4]();
        setupch <| chan(p);
        (*f)(p);
    }

    fn worktask5(setupch: opaquechan_5wordsz, fptr: opaque) {
        let f: *fn(port[wordsz5])  = unsafe::reinterpret_cast(fptr);
        let p = port[wordsz5]();
        setupch <| chan(p);
        (*f)(p);
    }

    let p = port[chan[T]]();
    let setupch = chan(p);
    let fptr = unsafe::reinterpret_cast(ptr::addr_of(f));

    let Tsz = sys::size_of[T]();
    let t =
        if Tsz == sys::size_of[wordsz1]() {
            let setupchptr = unsafe::reinterpret_cast(setupch);
            spawn worktask1(setupchptr, fptr)
        } else if (Tsz == sys::size_of[wordsz2]()) {
            let setupchptr = unsafe::reinterpret_cast(setupch);
            spawn worktask2(setupchptr, fptr)
        } else if (Tsz == sys::size_of[wordsz3]()) {
            let setupchptr = unsafe::reinterpret_cast(setupch);
            spawn worktask3(setupchptr, fptr)
        } else if (Tsz == sys::size_of[wordsz4]()) {
            let setupchptr = unsafe::reinterpret_cast(setupch);
            spawn worktask4(setupchptr, fptr)
        } else if (Tsz == sys::size_of[wordsz5]()) {
            let setupchptr = unsafe::reinterpret_cast(setupch);
            spawn worktask5(setupchptr, fptr)
        } else { fail #fmt("unhandled type size %u in task::worker", Tsz) };
    let ch;
    p |> ch;
    ret {task: t, chan: ch};
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
