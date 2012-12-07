// xfail-test
struct Foo { x: int }

impl Foo {
    fn stuff(&mut self) -> &self/mut Foo {
        return self;
    }
}

fn main() {
    let mut x = @mut Foo { x: 3 };
    x.stuff(); // error: internal compiler error: no enclosing scope with id 49
    // storing the result removes the error, so replacing the above
    // with the following, works:
    // let _y = x.stuff()

    // also making 'stuff()' not return anything fixes it
    // I guess the "dangling &ptr" cuases issues?
}
