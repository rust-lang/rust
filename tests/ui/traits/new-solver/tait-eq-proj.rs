// compile-flags: -Ztrait-solver=next
// check-pass

#![feature(type_alias_impl_trait)]

type Tait = impl Iterator<Item = impl Sized>;

/*

Consider the goal - AliasRelate(Tait, <[i32; 32] as IntoIterator>::IntoIter)
which is registered on the line above.

A. SubstRelate - fails (of course).

B. NormalizesToRhs - Tait normalizes-to <[i32; 32] as IntoIterator>::IntoIter
    * infer definition - Tait := <[i32; 32] as IntoIterator>::IntoIter

C. NormalizesToLhs - <[i32; 32] as IntoIterator>::IntoIter normalizes-to Tait
    * Find impl candidate, after substitute - std::array::IntoIter<i32, 32>
    * Equate std::array::IntoIter<i32, 32> and Tait
        * infer definition - Tait := std::array::IntoIter<i32, 32>

B and C are not equal, but they are equivalent modulo normalization.

We get around this by evaluating both the NormalizesToRhs and NormalizesToLhs
goals together. Essentially:
    A alias-relate B if A normalizes-to B and B normalizes-to A.

*/

fn a() {
    let _: Tait = IntoIterator::into_iter([0i32; 32]);
}

fn main() {}
