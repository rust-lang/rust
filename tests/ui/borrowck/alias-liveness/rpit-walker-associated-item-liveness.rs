//@ edition: 2024
//@ check-pass

// Minimized from the wax walker regression used by dolang and starbase in #161154.

#![forbid(unsafe_code)]
#![allow(dead_code)]

trait FileIterator: Iterator<Item = Self::Entry> {
    type Entry;
}

impl FileIterator for std::array::IntoIter<u8, 1> {
    type Entry = u8;
}

struct Glob<'t>(&'t str);

impl Glob<'_> {
    fn walk(&self) -> impl FileIterator<Entry = u8> + 'static {
        [0].into_iter()
    }
}

fn check(expression: &str) {
    let glob = Some(Glob(expression));
    let walker = match glob {
        Some(g) => g.walk(),
        None => Glob("tree").walk(),
    };
    for _ in walker {}
}

fn temporary() {
    let walker = Glob("tree").walk();
    for _ in walker {}
}

fn create_glob(expression: &str) -> Result<Glob<'_>, ()> {
    Ok(Glob(expression))
}

fn temporary_result(expression: &str) -> Result<(), ()> {
    let walker = create_glob(expression)?.walk();
    for _ in walker {}
    Ok(())
}

fn main() {}
