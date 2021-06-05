#[derive(Copy, Clone)]
struct Bar;

impl Copy for &'static mut Bar { }
//~^ ERROR the trait `Copy` may not be implemented for this type

fn main() {
}
