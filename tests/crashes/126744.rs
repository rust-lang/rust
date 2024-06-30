//@ known-bug: rust-lang/rust#126744
struct X {,}

fn main() {
    || {
        if let X { x: 1,} = (X {}) {}
    };
}
