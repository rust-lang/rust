// xfail-win32

// A port of task-killjoin to use a class with a dtor to manage
// the join.

use std;
import task;

struct notify {
    let ch: comm::Chan<bool>; let v: @mut bool;
    new(ch: comm::Chan<bool>, v: @mut bool) { self.ch = ch; self.v = v; }
    drop {
        error!("notify: task=%? v=%x unwinding=%b b=%b",
               task::get_task(),
               ptr::addr_of(*(self.v)) as uint,
               task::failing(),
               *(self.v));
        let b = *(self.v);
        comm::send(self.ch, b);
    }
}

fn joinable(+f: fn~()) -> comm::Port<bool> {
    fn wrapper(+c: comm::Chan<bool>, +f: fn()) {
        let b = @mut false;
        error!("wrapper: task=%? allocated v=%x",
               task::get_task(),
               ptr::addr_of(*b) as uint);
        let _r = notify(c, b);
        f();
        *b = true;
    }
    let p = comm::port();
    let c = comm::chan(p);
    do task::spawn_unlinked { wrapper(c, copy f) };
    p
}

fn join(port: comm::Port<bool>) -> bool {
    comm::recv(port)
}

fn supervised() {
    // Yield to make sure the supervisor joins before we
    // fail. This is currently not needed because the supervisor
    // runs first, but I can imagine that changing.
    error!("supervised task=%?", task::get_task);
    task::yield();
    fail;
}

fn supervisor() {
    error!("supervisor task=%?", task::get_task());
    let t = joinable(supervised);
    join(t);
}

fn main() {
    join(joinable(supervisor));
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
