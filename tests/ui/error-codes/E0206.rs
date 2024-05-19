#[derive(Copy, Clone)]
struct Bar;

impl Copy for &'static mut Bar { }
//~^ ERROR the trait `Copy` cannot be implemented for this type

fn main() {
}
