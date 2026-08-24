struct Bar {}

impl Bar {
    fn into_self(self) -> Bar {
        Bar(self)
        //~^ ERROR cannot find function, tuple struct or tuple variant `Bar` in this scope [E0423]
    }
}

fn main() {}
