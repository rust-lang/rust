pub trait Array {
    type Element;
}

pub trait Visit {
    fn visit() {}
}

impl Array for () {
    type Element = ();
}

impl<'a> Visit for () where
    (): Array<Element=&'a ()>,
{}

fn main() {
    <() as Visit>::visit(); //~ ERROR: type mismatch resolving
}
