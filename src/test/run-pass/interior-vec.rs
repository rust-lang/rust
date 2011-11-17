import rusti::vec_len;

#[abi = "rust-intrinsic"]
native mod rusti {
    fn vec_len<T>(&&v: [T]) -> uint;
}

fn main() unsafe {
    let v: [int] = [];
    assert (vec_len(v) == 0u); // zero-length
    let x = [1, 2];
    assert (vec_len(x) == 2u); // on stack
    let y = [1, 2, 3, 4, 5];
    assert (vec_len(y) == 5u); // on heap

    v += [];
    assert (vec_len(v) == 0u); // zero-length append
    x += [3];
    assert (vec_len(x) == 3u); // on-stack append
    y += [6, 7, 8, 9];
    assert (vec_len(y) == 9u); // on-heap append

    let vv = v + v;
    assert (vec_len(vv) == 0u); // zero-length add
    let xx = x + [4];
    assert (vec_len(xx) == 4u); // on-stack add
    let yy = y + [10, 11];
    assert (vec_len(yy) == 11u); // on-heap add
}
