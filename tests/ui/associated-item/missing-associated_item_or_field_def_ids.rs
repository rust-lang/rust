// Regression test for <https://github.com/rust-lang/rust/issues/137554>.

fn main() -> dyn Iterator + ?Iterator::advance_by(usize) {
    //~^ ERROR expected trait, found associated function `Iterator::advance_by`
    //~| ERROR relaxed bounds are not permitted in trait object types
    todo!()
}
