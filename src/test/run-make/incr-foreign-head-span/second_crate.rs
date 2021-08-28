extern crate first_crate;
use first_crate::OtherTrait;

#[cfg(not(second_run))]
trait Foo: OtherTrait {}

#[cfg(second_run)]
trait Bar: OtherTrait {}
