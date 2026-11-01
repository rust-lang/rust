// https://github.com/rust-lang/rust/issues/77919
fn main() {
    [1; <Multiply<Five, Five>>::VAL];
}
trait TypeVal<T> {
    const VAL: T;
}
struct Five;
struct Multiply<N, M> {
    _n: PhantomData, //~ ERROR cannot find type `PhantomData` in this scope
}
impl<N, M> TypeVal<usize> for Multiply<N, M> where N: TypeVal<VAL> {}
//~^ ERROR cannot find type `VAL` in this scope
//~| ERROR not all trait items implemented
