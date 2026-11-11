//@ check-fail

#![feature(specialization)]
#![allow(incomplete_features)]

// Tests that we don't overflow when using `default impl`.
// Regression test for #48515, #98478, and #117909.

// #48515

trait TypeString {
    fn type_string() -> &'static str;
}

default impl<T> TypeString for T {
    fn type_string() -> &'static str {
        "unknown type"
    }
}

impl TypeString for () {
    fn type_string() -> &'static str {
        "()"
    }
}

// #98478

trait Spam {}

trait SpamMore: Spam {}

default impl<T> Spam for T where T: SpamMore {}

struct A;

impl SpamMore for A {}
//~^ ERROR the trait bound `A: Spam` is not satisfied

fn needs_spam<T: Spam>() {}

// #117909

trait Set<T> {
    fn contains(&self, bit: T);
}

default impl<T, S> Set<&T> for S
where
    S: Set<T>,
{
    fn contains(&self, _: &T) {}
}

fn main() {
    let _ = <usize as TypeString>::type_string();
    //~^ ERROR the trait bound `usize: TypeString` is not satisfied

    needs_spam::<A>();
    //~^ ERROR the trait bound `A: Spam` is not satisfied

    0u32.contains(());
    //~^ ERROR no method named `contains` found for type `u32` in the current scope
}
