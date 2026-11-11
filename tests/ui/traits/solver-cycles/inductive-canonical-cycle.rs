//@ check-pass

// This test checks that we're correctly dealing with inductive cycles
// with canonical inference variables.

trait Trait<T, U> {}

trait IsNotU32 {}
impl IsNotU32 for i32 {}
impl<T: IsNotU32, U> Trait<T, U> for () // impl 1
where
    (): Trait<U, T>
{}

impl<T> Trait<u32, T> for () {} // impl 2

// If we now check whether `(): Trait<?0, ?1>` holds this has to
// result in ambiguity as both `for<T> (): Trait<u32, T>` and `(): Trait<i32, u32>`
// applies. The remainder of this test asserts that.

// If we were to error on inductive cycles with canonical inference variables
// this would be wrong:

// (): Trait<?0, ?1>
//  - impl 1
//      - ?0: IsNotU32 // ambig
//      - (): Trait<?1, ?0> // canonical cycle -> err
//      - ERR
//  - impl 2
//      - OK ?0 == u32
//
// Result: OK ?0 == u32.

// (): Trait<i32, u32>
//  - impl 1
//      - i32: IsNotU32 // ok
//      - (): Trait<u32, i32>
//          - impl 1
//              - u32: IsNotU32 // err
//              - ERR
//          - impl 2
//              - OK
//      - OK
//  - impl 2 (trivial ERR)
//
// Result OK

// This would mean that `(): Trait<?0, ?1>` is not complete,
// which is unsound if we're in coherence.

fn implements_trait<T, U>() -> (T, U)
where
    (): Trait<T, U>,
{
    todo!()
}

// A hack to only constrain the infer vars after first checking
// the `(): Trait<_, _>`.
trait Constrain<T> {}
impl<T> Constrain<T> for  T {}
fn constrain<T: Constrain<U>, U>(_: U) {}

fn main() {
    let (x, y) = implements_trait::<_, _>();

    constrain::<i32, _>(x);
    constrain::<u32, _>(y);
}
