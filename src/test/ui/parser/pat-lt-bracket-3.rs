struct Foo<T>(T, T);

impl<T> Foo<T> {
    fn foo(&self) {
        match *self {
            //~^ ERROR cannot move out of `self.0` which is behind a shared reference
            Foo<T>(x, y) => {
              println!("Goodbye, World!")
            }
        }
    }
}

fn main() {}
