// check-pass

fn test() -> impl Iterator<Item = impl Sized> {
    Box::new((0..).into_iter()) as Box<dyn Iterator<Item = _>>
}

fn main() {}
