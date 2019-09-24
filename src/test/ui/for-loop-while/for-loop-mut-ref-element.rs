// run-pass
// Tests that for loops can bind elements as mutable references

fn main() {
    for ref mut _a in std::iter::once(true) {}
}
