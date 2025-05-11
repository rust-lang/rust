//@ edition: 2021
//@ run-rustfix
#![feature(return_type_notation)]
#![allow(dead_code)]

trait Trait {
    async fn method() {}
}

fn foo<T: Trait<method(i32): Send>>() {}
//~^ ERROR argument types not allowed with return type notation

fn bar<T: Trait<method() -> (): Send>>() {}
//~^ ERROR return type not allowed with return type notation

fn baz<T: Trait<method(): Send>>() {}
//~^ ERROR return type notation arguments must be elided with `..`

fn foo_path<T: Trait>() where T::method(i32): Send {}
//~^ ERROR argument types not allowed with return type notation

fn bar_path<T: Trait>() where T::method() -> (): Send {}
//~^ ERROR return type not allowed with return type notation

fn bay_path<T: Trait>() where T::method(..) -> (): Send {}
//~^ ERROR return type not allowed with return type notation

fn baz_path<T: Trait>() where T::method(): Send {}
//~^ ERROR return type notation arguments must be elided with `..`

fn foo_qualified<T: Trait>() where <T as Trait>::method(i32): Send {}
//~^ ERROR expected associated type

fn bar_qualified<T: Trait>() where <T as Trait>::method() -> (): Send {}
//~^ ERROR expected associated type

fn baz_qualified<T: Trait>() where <T as Trait>::method(): Send {}
//~^ ERROR expected associated type

fn main() {}
