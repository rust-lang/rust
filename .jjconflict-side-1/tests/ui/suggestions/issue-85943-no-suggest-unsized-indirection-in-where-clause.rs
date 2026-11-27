// Regression test for #85943: should not emit suggestions for adding
// indirection to type parameters in where-clauses when suggesting
// adding `?Sized`.
struct A<T>(T) where T: Send;
struct B(A<[u8]>);
//~^ ERROR the size for values of type

pub fn main() {
}
