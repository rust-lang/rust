pub fn foo<'a: 'a, 'b: 'b, T: Clone, U: Clone, const N: usize>() {}

pub trait Trait<'b, T, const N: usize>: Sized {
    fn foo<'d: 'd, U, const M: bool>(self) {}
}

impl Trait<'static, i32, 1> for u8 {}
