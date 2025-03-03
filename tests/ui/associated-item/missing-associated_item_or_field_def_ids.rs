// Regression test for <https://github.com/rust-lang/rust/issues/137554>.

fn main() -> dyn Iterator + ?Iterator::advance_by(usize) {
    //~^ ERROR `?Trait` is not permitted in trait object types
    //~| ERROR expected trait, found associated function `Iterator::advance_by`
    //~| ERROR the value of the associated type `Item` in `Iterator` must be specified
    todo!()
}
