pub trait MyTrait {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}
