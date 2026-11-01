//@rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/alphabetical_or_trait_item_order
#![deny(clippy::arbitrary_source_item_ordering)]

// Trait definition order: b, a, c. Deliberately unordered to make the
// trait definition order differ from the alphabetical order; its own
// ordering check is not what this test is about.
#[allow(clippy::arbitrary_source_item_ordering)]
trait MixedOrder {
    fn b();
    fn a();
    fn c();
}

struct WrongInBothOrders;

// `c, a, b`: the pair (c, a) violates the alphabetical order, and in the
// trait definition `c` comes after `a`, so it violates both orderings.
impl MixedOrder for WrongInBothOrders {
    fn c() {}
    fn a() {}
    //~^ arbitrary_source_item_ordering
    fn b() {}
}

trait WithDefault {
    fn a() {}
    fn b() {}
    fn c() {}
}

struct SkipMiddleFailed;

impl WithDefault for SkipMiddleFailed {
    fn c() {}
    fn a() {}
    //~^ arbitrary_source_item_ordering
}

struct FullImplFailed;

impl WithDefault for FullImplFailed {
    fn c() {}
    fn a() {}
    //~^ arbitrary_source_item_ordering
    fn b() {}
}

trait WithDefault2 {
    fn a();
    fn b();
    fn c();
}

struct FullImplFailed2;

impl WithDefault2 for FullImplFailed2 {
    fn b() {}
    fn a() {}
    //~^ arbitrary_source_item_ordering
    fn c() {}
}

// Bare impls fall back to the alphabetical check.
struct BareImpl;

impl BareImpl {
    fn b() {}
    fn a() {}
    //~^ arbitrary_source_item_ordering
}

// Tests `const` in trait failed by invalid alphabetical order.
trait ConstTrait {
    const A: bool;
    const B: bool;
    const C: bool;
    fn method();
}

struct FullImplFailed3;

impl ConstTrait for FullImplFailed3 {
    const A: bool = true;
    const C: bool = true;
    const B: bool = false;
    //~^ arbitrary_source_item_ordering
    fn method() {}
}

trait ConstTrait2 {
    const A: bool = true;
    const B: bool = false;
    const C: bool = true;
    fn method() {}
}

struct SkipMiddleFailed2;

impl ConstTrait2 for SkipMiddleFailed2 {
    const C: bool = true;
    const A: bool = true;
    //~^ arbitrary_source_item_ordering
    fn method() {}
}

fn main() {}
