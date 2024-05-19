trait Foo {
    const BAR: u32;
}

struct SignedBar;

impl Foo for SignedBar {
    const BAR: i32 = -1;
    //~^ ERROR implemented const `BAR` has an incompatible type for trait [E0326]
}

fn main() {}
