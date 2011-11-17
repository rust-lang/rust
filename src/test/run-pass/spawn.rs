// -*- rust -*-

use std;

import std::task;

fn main() { let t = task::spawn_joinable(10, child); task::join(t); }

fn child(&&i: int) { log_err i; assert (i == 10); }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
