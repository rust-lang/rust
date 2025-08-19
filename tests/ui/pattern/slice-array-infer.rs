//@ check-pass

#![allow(unused_variables)]

struct Zeroes;
impl Into<&'static [usize; 3]> for Zeroes {
    fn into(self) -> &'static [usize; 3] {
        &[0; 3]
    }
}
impl Into<[usize; 3]> for Zeroes {
    fn into(self) -> [usize; 3] {
        [0; 3]
    }
}
fn main() {
    let [a, b, c] = Zeroes.into();
    let [d, e, f] = <Zeroes as Into<&'static [usize; 3]>>::into(Zeroes);
    let &[g, h, i] = Zeroes.into();
    let [j, k, l]: [usize; _] = Zeroes.into();
    let [m, n, o]: &[usize; _] = Zeroes.into();

    // check the binding mode of these patterns:
    let _: &[usize] = &[a, b, c, g, h, i, j, k, l];
    let _: &[&usize] = &[d, e, f, m, n, o];
}
