fn test(t: (i32, i32)) {}

struct Foo;

impl Foo {
    fn qux(&self) -> i32 {
        0
    }
}

fn bar() {
    let x = Foo;
    test(x.qux(), x.qux());
    //~^ ERROR this function takes 1 argument but 2 arguments were supplied
}

fn main() {}
