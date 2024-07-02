//@ known-bug: rust-lang/rust#126648
struct Outest(*const &'a ());

fn make() -> Outest {}

fn main() {
    if let Outest("foo") = make() {}
}
