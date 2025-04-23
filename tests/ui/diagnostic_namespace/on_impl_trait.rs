// used to ICE, see <https://github.com/rust-lang/rust/issues/130627>
// Instead it should just ignore the diagnostic attribute
#![feature(trait_alias)]

trait Test {}

#[diagnostic::on_unimplemented(message = "blah", label = "blah", note = "blah")]
//~^ WARN `#[diagnostic::on_unimplemented]` can only be applied to trait definitions
trait Alias = Test;

// Use trait alias as bound on type parameter.
fn foo<T: Alias>(v: &T) {}

pub fn main() {
    foo(&1);
    //~^ ERROR the trait bound `{integer}: Alias` is not satisfied
}
