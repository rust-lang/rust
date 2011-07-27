

type pair[T] = {x: T, y: T};

fn main() {
    let x: pair[int] = {x: 10, y: 12};
    assert (x.x == 10);
    assert (x.y == 12);
}