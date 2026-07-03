//@rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/trait_definition_order
//@check-pass
#![deny(clippy::arbitrary_source_item_ordering)]

struct Languages;
struct Language;
struct Iter<'a, T>(&'a [&'a T]);

impl<'a> Iterator for Iter<'a, Language> {
    type Item = &'a Language;
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'a> IntoIterator for &'a Languages {
    type Item = &'a Language;
    type IntoIter = Iter<'a, Language>;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

fn main() {}
