fn main() {
    fn echo<T>(c: int, x: &[T]) { }

    let y: fn(&[int]) = bind echo(42, _);

    y([1]);
}
