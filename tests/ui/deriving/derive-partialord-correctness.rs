// run-pass
// Original issue: #49650

#[derive(PartialOrd, PartialEq)]
struct FloatWrapper(f64);

fn main() {
    assert!((0.0 / 0.0 >= 0.0) == (FloatWrapper(0.0 / 0.0) >= FloatWrapper(0.0)))
}
