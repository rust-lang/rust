pub static FOO: u32 = FOO;
//~^ ERROR encountered static that tried to initialize itself with itself

#[derive(Copy, Clone)]
pub union Foo {
    x: u32,
}

pub static BAR: Foo = BAR;
//~^ ERROR encountered static that tried to initialize itself with itself

fn main() {}
