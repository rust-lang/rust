pub static FOO: u32 = FOO;
//~^ ERROR could not evaluate static initializer

#[derive(Copy, Clone)]
pub union Foo {
    x: u32,
}

pub static BAR: Foo = BAR;
//~^ ERROR could not evaluate static initializer

fn main() {}
