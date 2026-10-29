//@ run-pass


fn view<T>(x: &[T]) -> &[T] {x}

pub fn main() {
    let v = vec![1, 2, 3];
    let x = view(&v);
    let y = view(x);
    assert!((v[0] == x[0]) && (v[0] == y[0]));
}
