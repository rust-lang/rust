// xfail-stage0

/**
   Exercises task pinning and unpinning. Doesn't really ensure it
   works, just makes sure it runs.
*/

use std;

import std::task;

fn main() { task::pin(); task::unpin(); }