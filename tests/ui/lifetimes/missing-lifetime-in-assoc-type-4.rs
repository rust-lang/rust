struct S;
struct T;

impl IntoIterator for &S {
    type Item = &T;
    //~^ ERROR in the trait associated type
    type IntoIter<'a> = std::collections::btree_map::Values<'a, i32, T>;
    //~^ ERROR lifetime parameters or bounds on type `IntoIter` do not match the trait declaration

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}
fn main() {}
