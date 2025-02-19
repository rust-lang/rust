//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ reference: attributes.diagnostic.do_not_recommend.intro

#[diagnostic::on_unimplemented(message = "Very important message!")]
trait TheImportantOne {}

trait ImplementationDetail {
    type Restriction;
}

#[diagnostic::do_not_recommend]
impl<T: ImplementationDetail<Restriction = ()>> TheImportantOne for T {}

// Comment out this `impl` to show the expected error message.
impl ImplementationDetail for u8 {
    type Restriction = u8;
}

fn verify<T: TheImportantOne>() {}

pub fn main() {
    verify::<u8>();
    //~^ERROR: Very important message! [E0277]
}
