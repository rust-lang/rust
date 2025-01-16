#![feature(extern_types)]
#![feature(sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized};

fn needs_pointeesized<T: ?Sized + PointeeSized>() {}
fn needs_metasized<T: ?Sized + MetaSized>() {}
fn needs_sized<T: Sized>() {}

fn main() {
    needs_pointeesized::<u8>();
    needs_metasized::<u8>();
    needs_sized::<u8>();

    needs_pointeesized::<str>();
//~^ ERROR values of type `str` may or may not have a size
    needs_metasized::<str>();
//~^ ERROR the size for values of type `str` cannot be known
    needs_sized::<str>();
//~^ ERROR the size for values of type `str` cannot be known at compilation time

    extern "C" {
        type Foo;
    }

    needs_pointeesized::<Foo>();
//~^ ERROR values of type `main::Foo` may or may not have a size
    needs_metasized::<Foo>();
//~^ ERROR the size for values of type `main::Foo` cannot be known
    needs_sized::<Foo>();
//~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
}
