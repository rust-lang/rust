struct Foo;

impl<T: Default> Foo { //~ ERROR E0207
    fn get(&self) -> T {
        <T as Default>::default()
    }
}

fn main() {
}
