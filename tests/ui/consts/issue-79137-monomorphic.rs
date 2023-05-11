// check-pass

// Verify that variant count intrinsic can still evaluate for types like `Option<T>`.

#![feature(variant_count)]

pub struct GetVariantCount<T>(T);

impl<T> GetVariantCount<T> {
    pub const VALUE: usize = std::mem::variant_count::<T>();
}

const fn check_variant_count<T>() -> bool {
    matches!(GetVariantCount::<Option<T>>::VALUE, GetVariantCount::<Option<()>>::VALUE)
}

fn main() {
    assert!(check_variant_count::<()>());
}
