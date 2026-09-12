// Regression test for <https://github.com/rust-lang/rust/issues/137554>.

fn main() -> dyn Iterator + ?Iterator::advance_by(usize) {
    //~^ ERROR cannot find trait `advance_by` in trait `Iterator`
    //~| ERROR relaxed bounds are not permitted in trait object types
    todo!()
}
