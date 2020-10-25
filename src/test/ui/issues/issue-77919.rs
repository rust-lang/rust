fn main() {
    [1; <Multiply<Five, Five>>::VAL]; //~ ERROR evaluation of constant value failed
}
trait TypeVal<T> {
    const VAL: T; //~ ERROR any use of this value will cause an error
}
struct Five;
struct Multiply<N, M> {
    _n: PhantomData, //~ ERROR cannot find type `PhantomData` in this scope
}
impl<N, M> TypeVal<usize> for Multiply<N, M> where N: TypeVal<VAL> {}
//~^ ERROR cannot find type `VAL` in this scope
//~| ERROR not all trait items implemented, missing: `VAL`
