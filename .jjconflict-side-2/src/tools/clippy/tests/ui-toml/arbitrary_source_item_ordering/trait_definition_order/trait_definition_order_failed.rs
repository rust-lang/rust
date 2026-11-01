//@revisions: trait_def
//@rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/trait_definition_order
#![deny(clippy::arbitrary_source_item_ordering)]

// A trait whose definition order happens to be alphabetical.
trait AlphaOrder {
    fn a();
    fn b();
    fn c();
}

struct WrongInBothOrders;

// `a, c, b` violates both the alphabetical order and the trait
// definition order, so it is linted in every mode.
impl AlphaOrder for WrongInBothOrders {
    fn a() {}
    fn c() {}
    fn b() {}
    //~^ arbitrary_source_item_ordering
}

trait AlphaOrder2 {
    fn a() {}
    fn b() {}
    fn c() {}
}

struct WrongInBothOrder2;

impl AlphaOrder2 for WrongInBothOrder2 {
    fn a() {}
    fn c() {}
    fn b() {}
    //~^ arbitrary_source_item_ordering
}

// Bare impls have no trait definition to compare against, so the
// trait_item_ordering mode falls back to the alphabetical check.
struct BareImpl;
impl BareImpl {
    fn a() {}
    fn c() {}
    fn b() {}
    //~^ arbitrary_source_item_ordering
}

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
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
    type Item = &'a Language;
    //~^ arbitrary_source_item_ordering
    type IntoIter = Iter<'a, Language>;
}

// Tests `const` in trait when failed by invalid trait definition order (e.g. Not in alphabetical
// order).
trait ConstTrait {
    const A: bool;
    const B: bool;
    const C: bool;
    fn method();
}

struct WrongInBothOrder3;

impl ConstTrait for WrongInBothOrder3 {
    const A: bool = true;
    const C: bool = true;
    const B: bool = false;
    //~^ arbitrary_source_item_ordering
    fn method() {}
}

// Tests skipped trait which has `const` in trait failed by invalid trait definition order (e.g. Not
// in alphabetical order).
trait SkipConstTrait {
    const A: bool = true;
    const B: bool = false;
    const C: bool = true;
    fn method() {}
}

struct WrongInBothOrder4;

impl SkipConstTrait for WrongInBothOrder4 {
    const C: bool = true;
    const A: bool = true;
    //~^ arbitrary_source_item_ordering
    fn method() {}
}

fn main() {}
