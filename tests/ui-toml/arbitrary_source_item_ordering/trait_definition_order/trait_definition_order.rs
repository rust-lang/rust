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

// Default trait impl ordering.
trait WithDefault {
    fn a() {}
    fn b() {}
    fn c() {}
}

struct SkipMiddle;

impl WithDefault for SkipMiddle {
    fn a() {}
    fn c() {}
}

struct FullImpl;

impl WithDefault for FullImpl {
    fn a() {}
    fn b() {}
    fn c() {}
}

trait WithDefault2 {
    fn a();
    fn b();
    fn c();
}

struct FullImpl2;

impl WithDefault2 for FullImpl2 {
    fn a() {}
    fn b() {}
    fn c() {}
}

// Tests `const` in trait.
trait ConstTrait {
    const A: bool;
    const B: bool;
    const C: bool;
    fn method();
}

struct FullImpl3;

// Tests `const` in trait without value.
impl ConstTrait for FullImpl3 {
    const A: bool = true;
    const B: bool = false;
    const C: bool = true;
    fn method() {}
}

trait ConstTrait2 {
    const A: bool = true;
    const B: bool = false;
    const C: bool = true;
    fn method() {}
}

struct SkipImpl;

// Tests skipped `const` in trait.
impl ConstTrait2 for SkipImpl {
    const A: bool = true;
    const C: bool = true;
    fn method() {}
}

fn main() {}
