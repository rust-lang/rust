// -*- rust -*-

use std;

import std::task;

fn main() {
    let t = task::_spawn(bind child(10));
    task::join_id(t);
}

fn child(i: int) { log_err i; assert (i == 10); }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
