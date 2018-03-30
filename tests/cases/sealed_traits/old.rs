pub trait PublicTrait : private::PrivateTrait { }

mod private {
    pub trait PrivateTrait { }
}
