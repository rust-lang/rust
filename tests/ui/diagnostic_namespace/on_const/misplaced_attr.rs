#![feature(const_trait_impl, const_cmp, diagnostic_on_const)]
#![deny(misplaced_diagnostic_attributes)]

#[diagnostic::on_const(message = "tadaa", note = "boing")]
//~^ ERROR: cannot be used on
pub struct Foo;

#[diagnostic::on_const(message = "tadaa", note = "boing")]
//~^ ERROR: `#[diagnostic::on_const]` can only be applied to non-const trait implementations
const impl PartialEq for Foo {
    fn eq(&self, _other: &Foo) -> bool {
        true
    }
}

#[diagnostic::on_const(message = "tadaa", note = "boing")]
//~^ ERROR: cannot be used on
impl Foo {
    fn eq(&self, _other: &Foo) -> bool {
        true
    }
}

impl PartialOrd for Foo {
    #[diagnostic::on_const(message = "tadaa", note = "boing")]
    //~^ ERROR: cannot be used on
    fn partial_cmp(&self, other: &Foo) -> Option<std::cmp::Ordering> {
        None
    }
}

fn main() {}
