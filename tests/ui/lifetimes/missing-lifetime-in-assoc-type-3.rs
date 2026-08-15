struct S;
struct T;

impl IntoIterator for &S {
    type Item = &T;
    //~^ ERROR missing lifetime in associated type
    type IntoIter = std::collections::btree_map::Values<i32, T>;
    //~^ ERROR missing lifetime specifier

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}
fn main() {}
