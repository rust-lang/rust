pub trait MyIter {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}
