struct S;
struct T;

impl<'a> IntoIterator for &'_ S {
    //~^ ERROR E0207
    //~| NOTE there is a named lifetime specified on the impl block you could use
    //~| NOTE unconstrained lifetime parameter
    //~| HELP consider using the named lifetime here instead of an implict lifetime
    type Item = &T;
    //~^ ERROR in the trait associated type
    //~| HELP consider using the lifetime from the impl block
    //~| NOTE this lifetime must come from the implemented type
    type IntoIter = std::collections::btree_map::Values<'a, i32, T>;

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}
fn main() {}
