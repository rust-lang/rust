//~ NOTE in the trait the associated type is declared without lifetime parameters
struct S;
struct T;

trait Trait {
    type Item;
    type IntoIter;
    fn into_iter(self) -> Self::IntoIter;
}

impl<'a> Trait for &'_ S {
    //~^ ERROR E0207
    //~| NOTE there is a named lifetime specified on the impl block you could use
    //~| NOTE unconstrained lifetime parameter
    //~| HELP consider using the named lifetime here instead of an implict lifetime
    type Item = &T;
    //~^ ERROR missing lifetime in associated type
    //~| HELP consider using the lifetime from the impl block
    //~| NOTE this lifetime must come from the implemented type
    type IntoIter = std::collections::btree_map::Values<'a, i32, T>;

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}
fn main() {}
