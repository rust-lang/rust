// xfail-stage0

import rusti::ivec_len;

native "rust-intrinsic" mod rusti {
    fn ivec_len[T](v: &T[]) -> uint;
}

fn main() {
    let v: int[] = ~[];
    assert (ivec_len(v) == 0u); // zero-length
    let x = ~[1, 2];
    assert (ivec_len(x) == 2u); // on stack
    let y = ~[1, 2, 3, 4, 5];
    assert (ivec_len(y) == 5u); // on heap

    v += ~[];
    assert (ivec_len(v) == 0u); // zero-length append
    x += ~[3];
    assert (ivec_len(x) == 3u); // on-stack append
    y += ~[6, 7, 8, 9];
    assert (ivec_len(y) == 9u); // on-heap append

    let vv = v + v;
    assert (ivec_len(vv) == 0u); // zero-length add
    let xx = x + ~[4];
    assert (ivec_len(xx) == 4u); // on-stack add
    let yy = y + ~[10, 11];
    assert (ivec_len(yy) == 11u); // on-heap add
}

