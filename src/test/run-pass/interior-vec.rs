// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3

// works, but leaks in the compiler :(

import rusti::ivec_len;

native "rust-intrinsic" mod rusti {
    fn ivec_len[T](&T[] v) -> uint;
}

fn main() {
    let int[] v = ~[];
    assert (ivec_len(v) == 0u);     // zero-length
    auto x = ~[ 1, 2 ];
    assert (ivec_len(x) == 2u);     // on stack
    auto y = ~[ 1, 2, 3, 4, 5 ];
    assert (ivec_len(y) == 5u);     // on heap

    v += ~[];
    assert (ivec_len(v) == 0u);     // zero-length append
    x += ~[ 3 ];
    assert (ivec_len(x) == 3u);     // on-stack append
    y += ~[ 6, 7, 8, 9 ];
    assert (ivec_len(y) == 9u);     // on-heap append

    auto vv = v + v;
    assert (ivec_len(vv) == 0u);     // zero-length add
    auto xx = x + ~[ 4 ];
    assert (ivec_len(xx) == 4u);     // on-stack add
    auto yy = y + ~[ 10, 11 ];
    assert (ivec_len(yy) == 11u);    // on-heap add
}

