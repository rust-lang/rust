trait Dancer {
    fn dance(&self) -> _ {
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
        self.dance()
    }
}

fn main() {}
