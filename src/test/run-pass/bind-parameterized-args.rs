fn main() {
    fn echo[T](c: int, x: vec[T]) { }

    let y: fn(vec[int])  = bind echo(42, _);

    y([1]);
}
