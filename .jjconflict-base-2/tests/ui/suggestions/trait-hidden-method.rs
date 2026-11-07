// #107983 - testing that `__iterator_get_unchecked` isn't suggested
// HELP included so that compiletest errors on the bad suggestion
pub fn i_can_has_iterator() -> impl Iterator<Item = u32> {
    Box::new(1..=10) as Box<dyn Iterator>
    //~^ ERROR the value of the associated type `Item`
    //~| HELP specify the associated type
}

fn main() {}
