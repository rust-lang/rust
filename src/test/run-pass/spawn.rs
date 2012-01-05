// -*- rust -*-

use std;

import task;

fn main() {
    let t = task::spawn_joinable {|| child(10); };
    task::join(t);
}

fn child(&&i: int) { log(error, i); assert (i == 10); }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
