// -*- rust -*-

use std;
import task::spawn;

fn main() { spawn {|| child((10, 20, 30, 40, 50, 60, 70, 80, 90)); }; }

fn child(&&args: (int, int, int, int, int, int, int, int, int)) {
    let (i1, i2, i3, i4, i5, i6, i7, i8, i9) = args;
    log(error, i1);
    log(error, i2);
    log(error, i3);
    log(error, i4);
    log(error, i5);
    log(error, i6);
    log(error, i7);
    log(error, i8);
    log(error, i9);
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
// End:
