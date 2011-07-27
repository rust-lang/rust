
/*
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

use std;

import std::task::join;

fn grandchild(c: chan[int]) { c <| 42; }

fn child(c: chan[int]) {
    let _grandchild = spawn grandchild(c);
    join(_grandchild);
}

fn main() {
    let p: port[int] = port();

    let _child = spawn child(chan(p));

    let x: int;
    p |> x;

    log x;

    assert (x == 42);

    join(_child);
}