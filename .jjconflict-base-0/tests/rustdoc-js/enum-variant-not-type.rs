pub trait MyTrait {
    // Reduced from `arti` crate.
    // https://tpo.pages.torproject.net/core/doc/rust/tor_config/list_builder/trait.DirectDefaultEmptyListBuilderAccessors.html#associatedtype.T
    type T;
    fn not_appearing(&self) -> Option<&Self::T>;
}

pub fn my_fn<X>(t: X) -> X {
    t
}

pub trait AutoCorrectConfounder {
    type InsertUnnecessarilyLongTypeNameHere;
    fn assoc_type_acts_like_generic(
        &self,
        x: &Self::InsertUnnecessarilyLongTypeNameHere,
    ) -> Option<&Self::InsertUnnecessarilyLongTypeNameHere>;
}
