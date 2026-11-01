pub static FOO: u32 = FOO;
//~^ ERROR encountered static that tried to access itself during initialization

#[derive(Copy, Clone)]
pub union Foo {
    x: u32,
}

pub static BAR: Foo = BAR;
//~^ ERROR encountered static that tried to access itself during initialization

fn main() {}
