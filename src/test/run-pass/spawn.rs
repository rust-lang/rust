// -*- rust -*-

use std;


fn main() {
    task::spawn(|| child(10) );
}

fn child(&&i: int) { log(error, i); assert (i == 10); }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
