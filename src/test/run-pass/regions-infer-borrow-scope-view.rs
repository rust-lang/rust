fn view<T>(x: &r/[T]) -> &r/[T] {x}

fn main() {
    let v = ~[1, 2, 3];
    let x = view(v);
    let y = view(x);
    assert (v[0] == x[0]) && (v[0] == y[0]);
}

