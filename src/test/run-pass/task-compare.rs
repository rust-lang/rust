// xfail-stage1
// xfail-stage2
// xfail-stage3

/**
   A test case for issue #577, which also exposes #588
*/

use std;
import std::task::join;

fn child() { }

fn main() {
    // tasks
    let t1;
    let t2;

    t1 = spawn child();
    t2 = spawn child();

    assert (t1 == t1);
    assert (t1 != t2);

    // ports
    let p1;
    let p2;

    p1 = port[int]();
    p2 = port[int]();

    assert (p1 == p1);
    assert (p1 != p2);

    // channels
    let c1;
    let c2;

    c1 = chan(p1);
    c2 = chan(p2);

    assert (c1 == c1);
    assert (c1 != c2);

    join(t1);
    join(t2);
}