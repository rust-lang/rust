#![warn(clippy::type_id_on_box)]

use std::any::{Any, TypeId};
use std::ops::Deref;

trait AnySubTrait: Any {}
impl<T: Any> AnySubTrait for T {}

// `Any` is an indirect supertrait
trait AnySubSubTrait: AnySubTrait {}
impl<T: AnySubTrait> AnySubSubTrait for T {}

// This trait mentions `Any` in its predicates, but it is not a subtrait of `Any`.
trait NormalTrait
where
    i32: Any,
{
}
impl<T> NormalTrait for T {}

fn main() {
    // (currently we don't look deeper than one level into the supertrait hierarchy, but we probably
    // could)
    let b: Box<dyn AnySubSubTrait> = Box::new(1);
    let _ = b.type_id();
    //~^ type_id_on_box

    let b: Box<dyn NormalTrait> = Box::new(1);
    let _ = b.type_id();
    //~^ type_id_on_box
}
