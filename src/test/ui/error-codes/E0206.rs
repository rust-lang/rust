type Foo = [u8; 256];

impl Copy for Foo { }
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types

#[derive(Copy, Clone)]
struct Bar;

impl Copy for &'static mut Bar { }
//~^ ERROR the trait `Copy` may not be implemented for this type

fn main() {
}
