#![warn(clippy::multi_assignments)]
fn main() {
    let (mut a, mut b, mut c, mut d) = ((), (), (), ());
    a = b = c;
    a = b = c = d;
    a = b = { c };
    a = { b = c };
    a = (b = c);
}
