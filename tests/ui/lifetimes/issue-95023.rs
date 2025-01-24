struct ErrorKind;
struct Error(ErrorKind);
impl Fn(&isize) for Error {
    //~^ ERROR manual implementations of `Fn` are experimental [E0183]
    //~| ERROR associated item constraints are not allowed here [E0229]
    //~| ERROR not all trait items implemented
    //~| ERROR expected a `FnMut(&isize)` closure, found `Error`
    fn foo<const N: usize>(&self) -> Self::B<{ N }>;
    //~^ ERROR associated function in `impl` without body
    //~| ERROR method `foo` is not a member of trait `Fn` [E0407]
    //~| ERROR associated type `B` not found for `Self` [E0220]
}
fn main() {}
