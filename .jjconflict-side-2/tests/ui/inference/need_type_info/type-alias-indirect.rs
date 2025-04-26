// An addition to the `type-alias.rs` test,
// see the FIXME in that file for why this test
// exists.
//
// If there is none, feel free to remove this test
// again.
struct Ty<T>(T);
impl<T> Ty<T> {
    fn new() {}
}

type IndirectAlias<T> = Ty<Box<T>>;
fn indirect_alias() {
    IndirectAlias::new();
    //~^ ERROR type annotations needed
}

fn main() {}
