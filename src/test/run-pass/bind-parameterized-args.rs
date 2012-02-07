fn main() {
    fn echo<T>(c: int, x: [T]) { }

    let y: fn@([int]) = echo(42, _);

    y([1]);
}
