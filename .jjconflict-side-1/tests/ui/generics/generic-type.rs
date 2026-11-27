//@ run-pass



struct Pair<T> {x: T, y: T}

pub fn main() {
    let x: Pair<isize> = Pair {x: 10, y: 12};
    assert_eq!(x.x, 10);
    assert_eq!(x.y, 12);
}
