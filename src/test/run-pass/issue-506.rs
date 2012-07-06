/*
  A reduced test case for Issue #506, provided by Rob Arnold.

  Testing spawning foreign functions
*/

use std;
import task;

#[abi = "cdecl"]
extern mod rustrt {
    fn unsupervise();
}

fn main() { task::spawn(rustrt::unsupervise); }
