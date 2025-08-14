// Regression test for https://github.com/rust-lang/rust/issues/122549
// was fixed by https://github.com/rust-lang/rust/pull/122749

trait ConstChunksExactTrait<T> {
    fn const_chunks_exact<const N: usize>(&self) -> ConstChunksExact<'a, T, { N }>;
    //~^ ERROR undeclared lifetime
}

impl<T> ConstChunksExactTrait<T> for [T] {}
//~^ ERROR not all trait items implemented, missing: `const_chunks_exact`
struct ConstChunksExact<'rem, T: 'a, const N: usize> {}
//~^ ERROR use of undeclared lifetime name `'a`
//~^^ ERROR lifetime parameter
//~^^^ ERROR type parameter
impl<'a, T, const N: usize> Iterator for ConstChunksExact<'a, T, {}> {
//~^ ERROR the const parameter `N` is not constrained by the impl trait, self type, or predicates
//~^^ ERROR mismatched types
//~| ERROR missing: `next`
    type Item = &'a [T; N];
}

fn main() {
    let slice = &[1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let mut iter = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].iter();

    for a in slice.const_chunks_exact::<3>() {
        assert_eq!(a, iter.next().unwrap());
    }
}
