#![feature(min_generic_const_args)]
#![feature(adt_const_params, unsized_const_params)]
#[derive(PartialEq, Eq, std::marker::ConstParamTy)]
pub enum Enum<T> {
    Unit,
    Tuple(),
    Store(T),
}
pub mod module {
    pub use super::Enum::Store;
}
fn main() {
    type const _: Enum<()> = Enum::<()>::Unit::<()>;
    //~^ ERROR: type arguments are not allowed on unit variant `Unit` [E0109]
    type const _: Enum<()> = Enum::<()>::Tuple::<()>();
    //~^ ERROR: type arguments are not allowed on tuple variant `Tuple` [E0109]
    type const _: Enum<()> = self::<i32>::Enum::<()>::Store;
    //~^ ERROR: type arguments are not allowed on module `generic_args_on_enum_variant_segments_fail` [E0109]
}
