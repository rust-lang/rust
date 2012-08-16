/*
  A reduced test case for Issue #506, provided by Rob Arnold.

  Testing spawning foreign functions
*/

use std;
import task;

#[abi = "cdecl"]
extern mod rustrt {
    fn rust_dbg_do_nothing();
}

fn main() {
    task::spawn(rustrt::rust_dbg_do_nothing);
}
