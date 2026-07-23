//@rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/alphabetical_or_trait_item_order
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

// Follows the trait definition order (but not alphabetical order):
// accepted, because either ordering is allowed in this mode.
impl<'a> IntoIterator for &'a Languages {
    type Item = &'a Language;
    type IntoIter = Iter<'a, Language>;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

struct AlsoLanguages;

// Follows the alphabetical order (but not the trait definition order):
// also accepted in this mode.
impl<'a> IntoIterator for &'a AlsoLanguages {
    type IntoIter = Iter<'a, Language>;
    type Item = &'a Language;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

// Bare impls only have the alphabetical ordering to compare against.
struct BareImpl;
impl BareImpl {
    fn a() {}
    fn b() {}
    fn c() {}
}

// Default trait impl ordering.
trait WithDefault {
    fn a() {}
    fn b() {}
    fn c() {}
}

trait WithDefault2 {
    fn a();
    fn b();
    fn c();
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

impl WithDefault2 for FullImpl {
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

// Skips `B` `C`, keeps order of `A` and `C`.
impl ConstTrait2 for SkipImpl {
    const A: bool = true;
    fn method() {}
}

fn main() {}
