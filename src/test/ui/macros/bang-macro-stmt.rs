// check-pass

// Tests that we parse a bang macro
// as a statement when it occurs in the trailing expression position,
// which allows it to expand to a statement

fn main() {
    macro_rules! a {
        ($e:expr) => { $e; }
    }
    a!(true)
}
