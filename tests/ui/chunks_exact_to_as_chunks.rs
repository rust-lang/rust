#![warn(clippy::chunks_exact_to_as_chunks)]
#![allow(unused, clippy::redundant_closure_call)]

fn main() {
    let slice = [1, 2, 3, 4, 5, 6, 7, 8];

    // Should trigger lint - literal constant
    let mut it = slice.chunks_exact(4);
    //~^ chunks_exact_to_as_chunks
    for chunk in it {}

    // Should trigger lint - const value
    const CHUNK_SIZE: usize = 4;
    let mut it = slice.chunks_exact(CHUNK_SIZE);
    //~^ chunks_exact_to_as_chunks
    for chunk in it {}

    // Should NOT trigger - runtime value
    let size = 4;
    let mut it = slice.chunks_exact(size);
    for chunk in it {}

    // Should trigger lint - with remainder
    let mut it = slice.chunks_exact(3);
    //~^ chunks_exact_to_as_chunks
    for chunk in &mut it {}
    for e in it.remainder() {}

    // Should trigger - mutable variant
    let mut arr = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut it = arr.chunks_exact_mut(4);
    //~^ chunks_exact_to_as_chunks
    for chunk in it {}

    // Should NOT trigger - type must unify with another branch
    let condition = true;
    let y = 3;
    let _ = if condition {
        slice.chunks_exact(5)
    } else {
        slice.chunks_exact(y)
    };

    fn foo<const N: usize>(slice: &[u8]) {
        // Should trigger - passing const parameters directly is allowed

        let _ = slice.chunks_exact(N);
        //~^ chunks_exact_to_as_chunks
        let _ = slice.chunks_exact({
            //~^ chunks_exact_to_as_chunks
            const fn bar<const M: usize>() -> usize {
                M
            }
            bar::<5>()
        });

        // Should NOT trigger - expressions with const parameters are not allowed

        let _ = slice.chunks_exact(N * 2);
        let _ = slice.chunks_exact(size_of::<A<N>>());
        let _ = slice.chunks_exact(A::<N>::A);
        let _ = slice.chunks_exact((|| N)());
        let _ = slice.chunks_exact({
            const fn bar<const M: usize>() -> usize {
                M
            }
            bar::<N>()
        });
    }
    struct A<const N: usize>;
    impl<const N: usize> A<N> {
        const A: usize = N + 1;
    }

    trait Trait {
        const C: usize;
    }
    fn bar<T: Trait>(slice: &[u8]) {
        // Should NOT trigger - expressions in const arguments referencing generic parameters are not
        // allowed

        let _ = slice.chunks_exact(T::C);
        let _ = slice.chunks_exact(size_of::<T>());
    }
}
