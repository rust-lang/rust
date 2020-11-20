// The instrinsic call for variant count should exhaustively match on `tp_ty` and forbid
// `ty::Projection`, `ty::Opaque`, `ty::Param`, `ty::Bound`, `ty::Placeholder` and `ty::Infer`
// variant. This test checks that it will fail if it's too generic.

#![feature(variant_count)]

pub struct GetVariantCount<T>(T);

impl<T> GetVariantCount<T> {
    pub const VALUE: usize = std::mem::variant_count::<T>();
}

const fn check_variant_count<T>() -> bool {
    matches!(GetVariantCount::<T>::VALUE, GetVariantCount::<T>::VALUE)
    //~^ ERROR constant pattern depends on a generic parameter
    //~| ERROR constant pattern depends on a generic parameter
}

fn main() {
    assert!(check_variant_count::<Option<()>>());
}
