fn foo(_: u16) {}
fn foo1(_: f64) {}
fn foo2(_: i32) {}

fn main() {
    foo(1u8);
//~^ ERROR mismatched types
    foo1(2f32);
//~^ ERROR mismatched types
    foo2(3i16);
//~^ ERROR mismatched types
}
