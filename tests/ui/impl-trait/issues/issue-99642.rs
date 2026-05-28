//@ check-pass

fn test() -> impl Iterator<Item = impl Sized> {
    Box::new(0..) as Box<dyn Iterator<Item = _>>
}

fn main() {}
