#![feature(supertrait_item_shadowing, return_type_notation)]
#![deny(resolving_to_items_shadowing_supertrait_items)]
#![allow(dead_code)]

trait Base {
    type Assoc;
}
trait Sub: Base {
    type Assoc;
}
trait Child: Sub {}

fn path<T: Sub>(_: T::Assoc) {}
//~^ ERROR trait item `Assoc` from `Sub` shadows identically named item
fn equality<T: Child<Assoc = ()>>() {}
//~^ ERROR trait item `Assoc` from `Sub` shadows identically named item
fn bound<T: Child<Assoc: Copy>>() {}
//~^ ERROR trait item `Assoc` from `Sub` shadows identically named item

impl Base for () {
    type Assoc = u8;
}
impl Sub for () {
    type Assoc = u16;
}
trait Method: Sub {
    fn method(_: Self::Assoc);
    //~^ ERROR trait item `Assoc` from `Sub` shadows identically named item
}
impl Method for () {
    fn method(_: Self::Assoc) {}
    //~^ ERROR trait item `Assoc` from `Sub` shadows identically named item
}

fn explicit<T: Sub>(_: <T as Sub>::Assoc) {}
fn direct<T: Sub<Assoc = ()>>() {}
#[allow(resolving_to_items_shadowing_supertrait_items)]
fn allowed<T: Child<Assoc = ()>>(_: T::Assoc) {}

trait Generic<T>: Base {
    type Assoc;
}
fn ambiguous<T: Generic<u8> + Generic<u16>>(_: T::Assoc) {}
//~^ ERROR ambiguous associated type `Assoc` in bounds of `T`

// Multiple instantiations of a shadowed trait should produce only one note.
trait Parent<T> {
    type Assoc;
}
trait Winner: Parent<u8> + Parent<u16> {
    type Assoc;
}
fn repeated<T: Winner>(_: T::Assoc) {}
//~^ ERROR trait item `Assoc` from `Winner` shadows identically named item

trait RtnBase {
    fn run() -> impl Send;
}
trait RtnSub: RtnBase {
    fn run() -> impl Send;
}
trait RtnChild: RtnSub {}
fn rtn_path<T: RtnSub>()
where
    T::run(..): Send,
    //~^ ERROR trait item `run` from `RtnSub` shadows identically named item
{
}
fn rtn_constraint<T: RtnChild<run(..): Send>>() {}
//~^ ERROR trait item `run` from `RtnSub` shadows identically named item

fn main() {}
