fn double<T: copy>(a: T) -> ~[T] { return ~[a] + ~[a]; }

fn double_int(a: int) -> ~[int] { return ~[a] + ~[a]; }

fn main() {
    let mut d = double(1);
    assert (d[0] == 1);
    assert (d[1] == 1);

    d = double_int(1);
    assert (d[0] == 1);
    assert (d[1] == 1);
}

