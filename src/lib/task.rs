native "rust" mod rustrt {
    fn task_sleep(uint time_in_us);
    fn task_yield();
    fn task_join(task t) -> int;
    fn unsupervise();
    fn pin_task();
    fn unpin_task();
    fn clone_chan(*rust_chan c) -> *rust_chan;

    type rust_chan;
}

/**
 * Hints the scheduler to yield this task for a specified ammount of time.
 *
 * arg: time_in_us maximum number of microseconds to yield control for
 */
fn sleep(uint time_in_us) {
    ret rustrt::task_sleep(time_in_us);
}

fn yield() {
    ret rustrt::task_yield();
}

tag task_result {
    tr_success;
    tr_failure;
}

fn join(task t) -> task_result {
    alt (rustrt::task_join(t)) {
        0 { tr_success }
        _ { tr_failure }
    }
}

fn unsupervise() {
    ret rustrt::unsupervise();
}

fn pin() {
    rustrt::pin_task();
}

fn unpin() {
    rustrt::unpin_task();
}

fn clone_chan[T](chan[T] c) -> chan[T] {
    auto cloned = rustrt::clone_chan(unsafe::reinterpret_cast(c));
    ret unsafe::reinterpret_cast(cloned);
}

fn send[T](chan[T] c, &T v) {
    c <| v;
}

fn recv[T](port[T] p) -> T {
    auto v; p |> v; v
}

// Spawn a task and immediately return a channel for communicating to it
fn worker[T](fn(port[T]) f) -> rec(task task, chan[T] chan) {
    // FIXME: This is frighteningly unsafe and only works for
    // a few cases

    type opaque = int;

    // FIXME: This terrible hackery is because worktask can't currently
    // have type params
    type wordsz1 = int;
    type wordsz2 = rec(int a, int b);
    type wordsz3 = rec(int a, int b, int c);
    type wordsz4 = rec(int a, int b, int c, int d);
    type opaquechan_1wordsz = chan[chan[wordsz1]];
    type opaquechan_2wordsz = chan[chan[wordsz2]];
    type opaquechan_3wordsz = chan[chan[wordsz3]];
    type opaquechan_4wordsz = chan[chan[wordsz4]];

    fn worktask1(opaquechan_1wordsz setupch, opaque fptr) {
        let *fn(port[wordsz1]) f = unsafe::reinterpret_cast(fptr);
        auto p = port[wordsz1]();
        setupch <| chan(p);
        (*f)(p);
    }

    fn worktask2(opaquechan_2wordsz setupch, opaque fptr) {
        let *fn(port[wordsz2]) f = unsafe::reinterpret_cast(fptr);
        auto p = port[wordsz2]();
        setupch <| chan(p);
        (*f)(p);
    }

    fn worktask3(opaquechan_3wordsz setupch, opaque fptr) {
        let *fn(port[wordsz3]) f = unsafe::reinterpret_cast(fptr);
        auto p = port[wordsz3]();
        setupch <| chan(p);
        (*f)(p);
    }

    fn worktask4(opaquechan_4wordsz setupch, opaque fptr) {
        let *fn(port[wordsz4]) f = unsafe::reinterpret_cast(fptr);
        auto p = port[wordsz4]();
        setupch <| chan(p);
        (*f)(p);
    }

    auto p = port[chan[T]]();
    auto setupch = chan(p);
    auto fptr = unsafe::reinterpret_cast(ptr::addr_of(f));

    auto Tsz = sys::size_of[T]();
    auto t = if Tsz == sys::size_of[wordsz1]() {
        auto setupchptr = unsafe::reinterpret_cast(setupch);
        spawn worktask1(setupchptr, fptr)
    } else if Tsz == sys::size_of[wordsz2]() {
        auto setupchptr = unsafe::reinterpret_cast(setupch);
        spawn worktask2(setupchptr, fptr)
    } else if Tsz == sys::size_of[wordsz3]() {
        auto setupchptr = unsafe::reinterpret_cast(setupch);
        spawn worktask3(setupchptr, fptr)
    } else if Tsz == sys::size_of[wordsz4]() {
        auto setupchptr = unsafe::reinterpret_cast(setupch);
        spawn worktask4(setupchptr, fptr)
    } else {
        fail #fmt("unhandled type size %u in task::worker", Tsz)
    };
    auto ch; p |> ch;
    ret rec(task = t, chan = ch);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
