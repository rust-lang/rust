trait Bar {
    fn foo(&mut self) -> _ {
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
        Self::foo() //~ ERROR  this function takes 1 argument but 0 arguments were supplied
    }
}

fn main() {}
