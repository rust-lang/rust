/**
   A test case for issue #577, which also exposes #588
*/

// FIXME: This won't work until we can compare resources
// xfail-test

use std;
import task;
import task::join;
import comm;

fn child() { }

fn main() {
    // tasks
    let t1;
    let t2;

    let c1 = child, c2 = child;
    t1 = task::spawn_joinable(c1);
    t2 = task::spawn_joinable(c2);

    assert (t1 == t1);
    assert (t1 != t2);

    // ports
    let p1;
    let p2;

    p1 = comm::port::<int>();
    p2 = comm::port::<int>();

    assert (p1 == p1);
    assert (p1 != p2);

    // channels
    let c1 = comm::chan(p1);
    let c2 = comm::chan(p2);

    assert (c1 == c1);
    assert (c1 != c2);

    join(t1);
    join(t2);
}
