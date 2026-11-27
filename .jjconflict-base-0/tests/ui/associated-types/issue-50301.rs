// Tests that HRTBs are correctly accepted -- https://github.com/rust-lang/rust/issues/50301
//@ check-pass
trait Trait
where
    for<'a> &'a Self::IntoIter: IntoIterator<Item = u32>,
{
    type IntoIter;
    fn get(&self) -> Self::IntoIter;
}

struct Impl(Vec<u32>);

impl Trait for Impl {
    type IntoIter = ImplIntoIter;
    fn get(&self) -> Self::IntoIter {
        ImplIntoIter(self.0.clone())
    }
}

struct ImplIntoIter(Vec<u32>);

impl<'a> IntoIterator for &'a ImplIntoIter {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = std::iter::Cloned<std::slice::Iter<'a, u32>>;
    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter().cloned()
    }
}

fn main() {
}
