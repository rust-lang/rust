pub trait MyTrait {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

pub struct Empty;

impl MyTrait for Empty {
    type Item = ();
    fn next(&mut self) -> Option<()> {
        None
    }
}

pub struct Void;

impl MyTrait for Void {
    type Item = ();
    fn next(&mut self) -> Option<()> {
        Some(())
    }
}
