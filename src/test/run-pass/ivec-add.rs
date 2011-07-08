// xfail-stage0

fn double[T](&T a) -> T[] { ret ~[a] + ~[a]; }

fn double_int(int a) -> int[] { ret ~[a] + ~[a]; }

fn main() {
    auto d = double(1);
    assert (d.(0) == 1);
    assert (d.(1) == 1);

    d = double_int(1);
    assert (d.(0) == 1);
    assert (d.(1) == 1);
}

