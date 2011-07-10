// xfail-stage0

import rusti::ivec_len_2;

native "rust-intrinsic" mod rusti {
    fn ivec_len_2[T](&T[] v) -> uint;
}

fn main() {
    let int[] v = ~[];
    assert (ivec_len_2(v) == 0u);     // zero-length
    auto x = ~[ 1, 2 ];
    assert (ivec_len_2(x) == 2u);     // on stack
    auto y = ~[ 1, 2, 3, 4, 5 ];
    assert (ivec_len_2(y) == 5u);     // on heap

    v += ~[];
    assert (ivec_len_2(v) == 0u);     // zero-length append
    x += ~[ 3 ];
    assert (ivec_len_2(x) == 3u);     // on-stack append
    y += ~[ 6, 7, 8, 9 ];
    assert (ivec_len_2(y) == 9u);     // on-heap append

    auto vv = v + v;
    assert (ivec_len_2(vv) == 0u);     // zero-length add
    auto xx = x + ~[ 4 ];
    assert (ivec_len_2(xx) == 4u);     // on-stack add
    auto yy = y + ~[ 10, 11 ];
    assert (ivec_len_2(yy) == 11u);    // on-heap add
}

