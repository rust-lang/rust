// Regression test for #69789: rustc generated an invalid suggestion
// when `&` reference from `&mut` iterator is mutated.

fn main() {
    for item in &mut std::iter::empty::<&'static ()>() {
        //~^ NOTE this iterator yields `&` references
        *item = ();
        //~^ ERROR cannot assign
        //~| NOTE  cannot be written
    }
}
