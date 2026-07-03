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

// Bare impls fall back to the alphabetical check.
struct BareImpl;
impl BareImpl {
    fn b() {}
    fn a() {}
    //~^ arbitrary_source_item_ordering
}

fn main() {}
