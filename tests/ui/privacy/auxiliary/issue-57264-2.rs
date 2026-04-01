mod inner {
    pub struct PubUnnameable;

    impl PubUnnameable {
        pub fn pub_method(self) {}
    }
}

pub trait PubTraitWithSingleImplementor {}
impl PubTraitWithSingleImplementor for Option<inner::PubUnnameable> {}
