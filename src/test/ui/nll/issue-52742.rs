#![feature(in_band_lifetimes)]

struct Foo<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

struct Bar<'b> {
    z: &'b u32,
}

impl Foo<'_, '_> {
    fn take_bar(&mut self, b: Bar<'_>) {
        self.y = b.z
        //~^ ERROR
    }
}

fn main() {}
