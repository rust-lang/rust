enum Foo { Foo_(Bar) }
struct Bar { x: Bar }
//~^ ERROR E0072

fn main() {
}
