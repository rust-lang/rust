use std::ops::Add;

#[derive(Clone)]
struct Foo(Box<usize>);

impl Add for Foo {
    type Output = Foo;

    fn add(self, f: Foo) -> Foo {
        let Foo(i) = self;
        let Foo(j) = f;
        Foo(Box::new(*i + *j))
    }
}

fn main() {
    let x = Foo(Box::new(3));
    let _y = {x} + x.clone(); // the `{x}` forces a move to occur
    //~^ ERROR borrow of moved value: `x`
}
