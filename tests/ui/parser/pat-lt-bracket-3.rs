struct Foo<T>(T, T);

impl<T> Foo<T> {
    fn foo(&self) {
        match *self {
            Foo<T>(x, y) => { //~ ERROR generic args in patterns require the turbofish syntax
              println!("Goodbye, World!")
            }
        }
    }
}

fn main() {}
