/// This type is temporary and exists to cut down the bloat of further PR's
/// moving `struct Ty` from `rustc_middle` to `rustc_type_ir`.
pub type Ty<I> = <I as crate::interner::Interner>::Ty;
