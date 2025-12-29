//@ known-bug: #138660
enum A {
    V1(isize) = 1..=10,
    V0 = 1..=10,
}
const B: &'static [A] = &[A::V0, A::V1(111)];
fn main() {}
