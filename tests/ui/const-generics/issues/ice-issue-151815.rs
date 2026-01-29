//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// ICE when using generic_const_expr inside type in combination with map():
// closure's generic params were being substituted with parent function's args.

#[derive(Debug, Default, Clone, Copy)]
pub struct Num(i64);

pub struct Matrix<T, const R: usize, const C: usize> {
    pub entries: [[T; C]; R],
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
{
    pub fn default() -> Matrix<T, R, C> {
        let entries = [[T::default(); C]; R];
        Matrix { entries }
    }
}

pub struct SecretKey<const K: usize>
where
    [[Num; 2 * K]; K]: Sized,
{
    pub xs: Vec<Matrix<Num, { 2 * K }, K>>,
}

fn gen_mac<const K: usize>(msg_len: usize) -> SecretKey<K>
where
    [[Num; 2 * K]; K]: Sized,
{
    let xs = (0..2 * msg_len).map(|_| Matrix::<Num, { 2 * K }, K>::default()).collect();
    SecretKey { xs }
}

fn main() {
    let msg_len = 128;
    let sk: SecretKey<2> = gen_mac(msg_len);
    println!("{:?}", sk.xs[0].entries[0][0].0);
}
