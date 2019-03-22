trait Foo {
    pub const Foo: u32;
    //~^ ERROR unnecessary visibility qualifier
}

fn main() {}
