// check-pass
// revisions: old next
//[next] compile-flags: -Ztrait-solver=next

// If we use canonical goals during trait solving we have to reevaluate
// the root goal of a cycle until we hit a fixpoint.
//
// Here `main` has a goal `(?0, ?1): Trait` which is canonicalized to
// `exists<^0, ^1> (^0, ^1): Trait`.
//
// - `exists<^0, ^1> (^0, ^1): Trait` -instantiate-> `(?0, ?1): Trait`
//   -`(?1, ?0): Trait` -canonicalize-> `exists<^0, ^1> (^0, ^1): Trait`
//     - COINDUCTIVE CYCLE OK (no constraints)
//   - `(): ConstrainToU32<?0>` -canonicalize-> `exists<^0> (): ConstrainToU32<^0>`
//     - OK (^0 = u32 -apply-> ?0 = u32)
//   - OK (?0 = u32 -canonicalize-> ^0 = u32)
//   - coinductive cycle with provisional result != final result, rerun
//
// - `exists<^0, ^1> (^0, ^1): Trait` -instantiate-> `(?0, ?1): Trait`
//   -`(?1, ?0): Trait` -canonicalize-> `exists<^0, ^1> (^0, ^1): Trait`
//     - COINDUCTIVE CYCLE OK (^0 = u32 -apply-> ?1 = u32)
//   - `(): ConstrainToU32<?0>` -canonicalize-> `exists<^0> (): ConstrainToU32<^0>`
//     - OK (^0 = u32 -apply-> ?1 = u32)
//   - OK (?0 = u32, ?1 = u32 -canonicalize-> ^0 = u32, ^1 = u32)
//   - coinductive cycle with provisional result != final result, rerun
//
// - `exists<^0, ^1> (^0, ^1): Trait` -instantiate-> `(?0, ?1): Trait`
//   -`(?1, ?0): Trait` -canonicalize-> `exists<^0, ^1> (^0, ^1): Trait`
//     - COINDUCTIVE CYCLE OK (^0 = u32, ^1 = u32 -apply-> ?1 = u32, ?0 = u32)
//   - `(): ConstrainToU32<?0>` -canonicalize-> `exists<^0> (): ConstrainToU32<^0>`
//     - OK (^0 = u32 -apply-> ?1 = u32)
//   - OK (?0 = u32, ?1 = u32 -canonicalize-> ^0 = u32, ^1 = u32)
//   - coinductive cycle with provisional result == final result, DONE
#![feature(rustc_attrs)]
#[rustc_coinductive]
trait Trait {}

impl<T, U> Trait for (T, U)
where
    (U, T): Trait,
    (): ConstrainToU32<T>,
{}

trait ConstrainToU32<T> {}
impl ConstrainToU32<u32> for () {}

fn impls_trait<T, U>()
where
    (T, U): Trait,
{}

fn main() {
    impls_trait::<_, _>();
}
