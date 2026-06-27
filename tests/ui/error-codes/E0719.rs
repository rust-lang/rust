//@check-pass
// TODO: Do I just remove this test?

type Unit = ();

fn test() -> Box<dyn Iterator<Item = (), Item = Unit>> {
    Box::new(None.into_iter())
}

fn main() {
    let _: &dyn Iterator<Item = i32, Item = i32>;
    test();
}
