// originally from glacier fixed/77919.rs
// encountered errors resolving bounds after type-checking

trait TypeVal<T> {
    const VAL: T;
}
struct Five;
struct Multiply<N, M> {
    _n: PhantomData,
}
impl<N, M> TypeVal<usize> for Multiply<N, M> where N: TypeVal<VAL> {}

fn main() {
    [1; <Multiply<Five, Five>>::VAL];
}
