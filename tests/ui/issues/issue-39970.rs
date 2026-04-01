trait Array<'a> {
    type Element: 'a;
}

trait Visit {
    fn visit() {}
}

impl<'a> Array<'a> for () {
    type Element = &'a ();
}

impl Visit for () where
    //(): for<'a> Array<'a, Element=&'a ()>, // No ICE
    (): for<'a> Array<'a, Element=()>, // ICE
{}

fn main() {
    <() as Visit>::visit();
    //~^ ERROR type mismatch resolving `<() as Array<'a>>::Element == ()`
}
