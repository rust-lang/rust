// issue rust-lang/rust#111667
// ICE failed to resolve instance for <[f32; 2] as CrossProduct ..
//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait CrossProduct<'a, T, R> {
    fn cross(&'a self, t: &'a T) -> R;
}

impl<'a, T, U, const N: usize> CrossProduct<'a, [U; N], [(&'a T, &'a U); N * N]> for [T; N] {
    fn cross(&'a self, us: &'a [U; N]) -> [(&'a T, &'a U); N * N] {
        std::array::from_fn(|i| (&self[i / N], &us[i % N]))
    }
}

pub fn main() {}
