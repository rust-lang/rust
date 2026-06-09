// Test that `variant_count` only gets evaluated once the type is concrete enough.

#![feature(variant_count)]

pub struct GetVariantCount<T>(T);

impl<T> GetVariantCount<T> {
    pub const VALUE: usize = std::mem::variant_count::<T>();
}

const fn check_variant_count<T>() -> bool {
    matches!(GetVariantCount::<T>::VALUE, GetVariantCount::<T>::VALUE)
    //~^ ERROR constant pattern cannot depend on generic parameters
}

fn main() {
    assert!(check_variant_count::<Option<()>>());
}
