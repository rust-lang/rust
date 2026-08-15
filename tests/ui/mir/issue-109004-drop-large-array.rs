//@ check-pass

const SZ: usize = 64_000_000;
type BigDrop = [String; SZ];

fn f(_dropme: BigDrop) {}

fn f2(_moveme: BigDrop) -> String {
    let [a, ..] = _moveme;
    a
}

fn main() {
    f(std::array::from_fn(|_| String::new()));
    f2(std::array::from_fn(|_| String::new()));
}
