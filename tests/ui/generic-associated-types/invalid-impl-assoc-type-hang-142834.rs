// Regression test for <https://github.com/rust-lang/rust/issues/142834>.

trait Trait {}
struct W<T>(T);
impl<T, U> Trait for W
//~^ ERROR missing generics for struct `W`
where
    W<T>: Trait,
    W<U>: Trait,
{
    type NewAssoc<T, U>
    //~^ ERROR the name `T` is already used for a generic parameter
    //~| ERROR the name `U` is already used for a generic parameter
    //~| ERROR type `NewAssoc` is not a member of trait `Trait`
        = (&'a (), &'b ())
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| ERROR use of undeclared lifetime name `'b`
    where
        W<U>: Trait;
}
fn main() {
}
