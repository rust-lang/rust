pub trait PublicTrait : private::PrivateTrait {
    fn abc(&self) -> bool;
}

mod private {
    pub trait PrivateTrait { }
}
