#![feature(extern_types)]
#![feature(sized_hierarchy)]

use std::marker::{SizeOfVal, PointeeSized};

fn needs_pointeesized<T: PointeeSized>() {}
fn needs_sizeofval<T: SizeOfVal>() {}
fn needs_sized<T: Sized>() {}

fn main() {
    needs_pointeesized::<u8>();
    needs_sizeofval::<u8>();
    needs_sized::<u8>();

    needs_pointeesized::<str>();
    needs_sizeofval::<str>();
    needs_sized::<str>();
//~^ ERROR the size for values of type `str` cannot be known at compilation time

    extern "C" {
        type Foo;
    }

    needs_pointeesized::<Foo>();
    needs_sizeofval::<Foo>();
//~^ ERROR the size for values of type `main::Foo` cannot be known
    needs_sized::<Foo>();
//~^ ERROR the size for values of type `main::Foo` cannot be known at compilation time
}
