fn double[T](a: &T) -> T[] { ret ~[a] + ~[a]; }

fn double_int(a: int) -> int[] { ret ~[a] + ~[a]; }

fn main() {
    let d = double(1);
    assert (d.(0) == 1);
    assert (d.(1) == 1);

    d = double_int(1);
    assert (d.(0) == 1);
    assert (d.(1) == 1);
}

