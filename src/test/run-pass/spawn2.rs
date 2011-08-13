// -*- rust -*-

use std;
import std::task::_spawn;

fn main() { _spawn(bind child(10, 20, 30, 40, 50, 60, 70, 80, 90)); }

fn child(i1: int, i2: int, i3: int, i4: int, i5: int, i6: int, i7: int,
         i8: int, i9: int) {
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
