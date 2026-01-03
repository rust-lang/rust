#![feature(const_trait_impl, const_cmp)]
#![deny(misplaced_diagnostic_attributes)]

#[diagnostic::on_const(message = "tadaa", note = "boing")]
//~^ ERROR: `#[diagnostic::on_const]` can only be applied to trait impls
pub struct Foo;

#[diagnostic::on_const(message = "tadaa", note = "boing")]
//~^ ERROR: `#[diagnostic::on_const]` can only be applied to trait impls
impl const PartialEq for Foo {
    fn eq(&self, _other: &Foo) -> bool {
        true
    }
}

#[diagnostic::on_const(message = "tadaa", note = "boing")]
//~^ ERROR: `#[diagnostic::on_const]` can only be applied to trait impls
impl Foo {
    fn eq(&self, _other: &Foo) -> bool {
        true
    }
}

impl PartialOrd for Foo {
    #[diagnostic::on_const(message = "tadaa", note = "boing")]
    //~^ ERROR: `#[diagnostic::on_const]` can only be applied to trait impls
    fn partial_cmp(&self, other: &Foo) -> Option<std::cmp::Ordering> {
        None
    }
}

fn main() {}
