//! Ambiguity errors blaming the same inference variable are merged into a single
//! diagnostic that mentions every unsatisfied requirement, instead of only the
//! first one while the others get canceled as tainted-by-error duplicates.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/103911>.

trait Trait {}
impl Trait for String {}
struct NotDefault;
impl Trait for NotDefault {}

fn as_input<T: Trait>(_: T) {}
fn constrained<T: Trait + Clone>(_: T) {}

fn two_bounds() {
    as_input(Default::default());
    //~^ ERROR type annotations needed
}

fn three_bounds() {
    constrained(Default::default());
    //~^ ERROR type annotations needed
}

fn main() {}
