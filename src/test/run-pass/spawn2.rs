// -*- rust -*-

use std;
import std::task::spawn;

fn main() { spawn((10, 20, 30, 40, 50, 60, 70, 80, 90), child); }

fn child(&&args: (int, int, int, int, int, int, int, int, int)) {
    let (i1, i2, i3, i4, i5, i6, i7, i8, i9) = args;
    log_err i1;
    log_err i2;
    log_err i3;
    log_err i4;
    log_err i5;
    log_err i6;
    log_err i7;
    log_err i8;
    log_err i9;
    assert (i1 == 10);
    assert (i2 == 20);
    assert (i3 == 30);
    assert (i4 == 40);
    assert (i5 == 50);
    assert (i6 == 60);
    assert (i7 == 70);
    assert (i8 == 80);
    assert (i9 == 90);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
