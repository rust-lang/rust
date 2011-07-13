


// -*- rust -*-
fn ack(int m, int n) -> int {
    if (m == 0) {
        ret n + 1;
    } else {
        if (n == 0) {
            ret ack(m - 1, 1);
        } else { ret ack(m - 1, ack(m, n - 1)); }
    }
}

fn main() {
    assert (ack(0, 0) == 1);
    assert (ack(3, 2) == 29);
    assert (ack(3, 4) == 125);
    // This takes a while; but a comparison may amuse: on win32 at least, the
    // posted C version of the 'benchmark' running ack(4,1) overruns its stack
    // segment and crashes. We just grow our stack (to 4mb) as we go.

    // assert (ack(4,1) == 65533);

}