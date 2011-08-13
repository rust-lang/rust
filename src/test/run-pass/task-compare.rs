/**
   A test case for issue #577, which also exposes #588
*/

use std;
import std::task;
import std::task::join_id;
import std::comm;

fn child() { }

fn main() {
    // tasks
    let t1;
    let t2;

    t1 = task::_spawn(bind child());
    t2 = task::_spawn(bind child());

    assert (t1 == t1);
    assert (t1 != t2);

    // ports
    let p1;
    let p2;

    p1 = comm::mk_port[int]();
    p2 = comm::mk_port[int]();

    assert (p1 == p1);
    assert (p1 != p2);

    // channels
    let c1;
    let c2;

    c1 = p1.mk_chan();
    c2 = p2.mk_chan();

    assert (c1 == c1);
    assert (c1 != c2);

    join_id(t1);
    join_id(t2);
}