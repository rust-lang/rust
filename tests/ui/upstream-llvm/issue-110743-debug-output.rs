// this test tracks superfluous debug output from LLVM which we can't control

// build-pass
// min-llvm-version: 16
// known-bug: #110743

const SZ: usize = 64_000_000;
type BigDrop = [String; SZ];

fn f(from_fn: BigDrop) {}

fn f2(_moveme: BigDrop) -> String {
    let [a, ..] = _moveme;
    a
}

fn main() {
    f(std::array::from_fn(|_| String::new()));
    f2(std::array::from_fn(|_| String::new()));
}
