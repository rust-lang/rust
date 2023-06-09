// Regression test for #48132. This was failing due to problems around
// the projection caching and dropck type enumeration.

// check-pass

pub struct Container<T: Iterator> {
    value: Option<T::Item>,
}

impl<T: Iterator> Container<T> {
    pub fn new(iter: T) -> Self {
        panic!()
    }
}

pub struct Wrapper<'a> {
    content: &'a Content,
}

impl<'a, 'de> Wrapper<'a> {
    pub fn new(content: &'a Content) -> Self {
        Wrapper {
            content: content,
        }
    }
}

pub struct Content;

fn crash_it(content: Content) {
    let items = vec![content];
    let map = items.iter().map(|ref o| Wrapper::new(o));

    let mut map_visitor = Container::new(map);

}

fn main() {}
