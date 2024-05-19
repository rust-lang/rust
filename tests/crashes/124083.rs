//@ known-bug: #124083

struct Outest(&'a ());

fn make() -> Outest {}

fn main() {
    if let Outest("foo") = make() {}
}
