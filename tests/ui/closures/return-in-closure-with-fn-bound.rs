// Regression test for https://github.com/rust-lang/rust/issues/155893.

fn func(_f: impl Fn()) {
    func(|| return 2)
    //~^ ERROR mismatched types
}

fn main() {}
