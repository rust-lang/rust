//@ known-bug: #139570
fn main() {
    |(1, 42), ()| yield;
}
