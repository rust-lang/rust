//@ run-rustfix
// https://github.com/rust-lang/rust/issues/79076

#[derive(Clone, Eq)] //~ ERROR [E0277]
pub struct Struct<T>(T);

impl<T: Clone, U> PartialEq<U> for Struct<T>
where
    U: Into<Struct<T>> + Clone
{
    fn eq(&self, _other: &U) -> bool {
        todo!()
    }
}

fn main() {}
