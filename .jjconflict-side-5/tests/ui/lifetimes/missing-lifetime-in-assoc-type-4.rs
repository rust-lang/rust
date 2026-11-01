struct S;
struct T;

impl IntoIterator for &S {
    type Item = &T;
    //~^ ERROR missing lifetime in associated type
    type IntoIter<'a> = std::collections::btree_map::Values<'a, i32, T>;
    //~^ ERROR lifetime parameters or bounds on associated type `IntoIter` do not match the trait declaration

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}
fn main() {}
