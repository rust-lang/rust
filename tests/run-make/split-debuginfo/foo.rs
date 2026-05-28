pub struct Foo {
    x: u32,
}

impl Foo {
    pub fn print(&self) {
        println!("{}", self.x);
    }
}

pub fn make_foo(x: u32) -> Foo {
    Foo { x }
}

fn main() {}
