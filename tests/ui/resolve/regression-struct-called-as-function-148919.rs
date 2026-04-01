struct Bar {}

impl Bar {
    fn into_self(self) -> Bar {
        Bar(self)
        //~^ ERROR expected function, tuple struct or tuple variant, found struct `Bar` [E0423]
    }
}

fn main() {}
