//@ check-pass

struct It;

struct Data {
    items: Vec<It>,
}

impl Data {
    fn new() -> Self {
        Self {
            items: vec![It, It],
        }
    }

    fn content(&self) -> impl Iterator<Item = &It> {
        self.items.iter()
    }
}

struct Container<'a> {
    name: String,
    resolver: Box<dyn Resolver + 'a>,
}

impl<'a> Container<'a> {
    fn new<R: Resolver + 'a>(name: &str, resolver: R) -> Self {
        Self {
            name: name.to_owned(),
            resolver: Box::new(resolver),
        }
    }
}

trait Resolver {}

impl<R: Resolver> Resolver for &R {}

impl Resolver for It {}

fn get<'a>(mut items: impl Iterator<Item = &'a It>) -> impl Resolver + 'a {
    items.next().unwrap()
}

fn get2<'a, 'b: 'b>(mut items: impl Iterator<Item = &'a It>) -> impl Resolver + 'a {
    items.next().unwrap()
}

fn main() {
    let data = Data::new();
    let resolver = get(data.content());

    let _ = ["a", "b"]
        .iter()
        .map(|&n| Container::new(n, &resolver))
        .map(|c| c.name)
        .collect::<Vec<_>>();

    let resolver = get2(data.content());

    let _ = ["a", "b"]
        .iter()
        .map(|&n| Container::new(n, &resolver))
        .map(|c| c.name)
        .collect::<Vec<_>>();
}
