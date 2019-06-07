#![crate_name = "foo"]
#![feature(const_generics)]

pub trait Array {
    type Item;
}

// @has foo/trait.Array.html
// @has - '//h3[@class="impl"]' 'impl<T, const N: usize> Array for [T; N]'
impl <T, const N: usize> Array for [T; N] {
    type Item = T;
}
