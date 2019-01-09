struct Foo<T>(T, T);

impl<T> Foo<T> {
    fn foo(&self) {
        match *self {
            Foo<T>(x, y) => {
            //~^ error: expected one of `=>`, `@`, `if`, or `|`, found `<`
              println!("Goodbye, World!")
            }
        }
    }
}

fn main() {}
