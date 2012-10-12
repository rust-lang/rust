// xfail-test
/**
   A test case for issue #577, which also exposes #588
*/

extern mod std;

fn child() { }

struct notify {
    ch: comm::Chan<bool>, v: @mut bool,
    drop {
        error!("notify: task=%? v=%x unwinding=%b b=%b",
               task::get_task(),
               ptr::addr_of(&(*(self.v))) as uint,
               task::failing(),
               *(self.v));
        let b = *(self.v);
        comm::send(self.ch, b);
    }
}

fn notify(ch: comm::Chan<bool>, v: @mut bool) -> notify {
    notify {
        ch: ch,
        v: v
    }
}

fn joinable(+f: fn~()) -> comm::Port<bool> {
    fn wrapper(+c: comm::Chan<bool>, +f: fn()) {
        let b = @mut false;
        error!("wrapper: task=%? allocated v=%x",
               task::get_task(),
               ptr::addr_of(&(*b)) as uint);
        let _r = notify(c, b);
        f();
        *b = true;
    }
    let p = comm::Port();
    let c = comm::Chan(&p);
    do task::spawn_unlinked { wrapper(c, copy f) };
    p
}

fn join(port: comm::Port<bool>) -> bool {
    comm::recv(port)
}

fn main() {
    // tasks
    let t1;
    let t2;

    let c1 = child, c2 = child;
    t1 = joinable(c1);
    t2 = joinable(c2);

    assert (t1 == t1);
    assert (t1 != t2);

    // ports
    let p1;
    let p2;

    p1 = comm::Port::<int>();
    p2 = comm::Port::<int>();

    assert (p1 == p1);
    assert (p1 != p2);

    // channels
    let c1 = comm::Chan(p1);
    let c2 = comm::Chan(p2);

    assert (c1 == c1);
    assert (c1 != c2);

    join(t1);
    join(t2);
}
