impl<T> Option<T> {
//~^ ERROR cannot define inherent `impl` for a type outside of the crate where the type is defined
    pub fn foo(&self) { }
}

fn main() { }
