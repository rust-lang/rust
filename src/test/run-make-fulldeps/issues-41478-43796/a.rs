#![crate_type = "lib"]
pub struct V<S>(#[allow(unused_tuple_struct_fields)] S);
pub trait An {
    type U;
}
pub trait F<A> {
}
impl<A: An> F<A> for V<<A as An>::U> {
}
