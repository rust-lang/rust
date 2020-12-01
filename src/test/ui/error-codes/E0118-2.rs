struct Foo;

impl &mut Foo {
    //~^ ERROR E0118
    fn bar(self) {}
}

fn main() {}
