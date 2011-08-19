// Sanity-check the code examples that appear in the object system
// documentation.
use std;
import std::comm::_chan;
import std::comm::send;
import std::comm::mk_port;

fn main() {

    // Ref.Item.Obj
    obj counter(state: @mutable int) {
        fn incr() { *state += 1; }
        fn get() -> int { ret *state; }
    }

    let c: counter = counter(@mutable 1);

    c.incr();
    c.incr();
    assert (c.get() == 3);

    obj my_obj() {
        fn get() -> int { ret 3; }
        fn foo() -> int {
            let c = self.get();
            ret c + 2; // returns 5
        }
    }

    let o = my_obj();
    assert (o.foo() == 5);

    // Ref.Type.Obj
    type taker =
        obj {
            fn take(int);
        };

    obj adder(x: @mutable int) {
        fn take(y: int) { *x += y; }
    }

    obj sender(c: _chan<int>) {
        fn take(z: int) { send(c, z); }
    }

    fn give_ints(t: taker) { t.take(1); t.take(2); t.take(3); }

    let p = mk_port::<int>();

    let t1: taker = adder(@mutable 0);
    let t2: taker = sender(p.mk_chan());

    give_ints(t1);
    give_ints(t2);
}

