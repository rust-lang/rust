// originally from glacier fixed/77919.rs
// encountered errors resolving bounds after type-checking
//@no-rustfix
trait TypeVal<T> {
    const VAL: T;
}
struct Five;
struct Multiply<N, M> {
    _n: PhantomData,
    //~^ ERROR: cannot find type
}
impl<N, M> TypeVal<usize> for Multiply<N, M> where N: TypeVal<VAL> {}
//~^ ERROR: cannot find type
//~| ERROR: not all trait items

fn main() {
    [1; <Multiply<Five, Five>>::VAL];
}
