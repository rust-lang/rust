//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for trait-system-refactor-initiative#205. Avoid constraining
// the opaque type when applying blanket impls.

trait Trait<T> {}

impl<T> Trait<T> for T {}
impl Trait<u32> for u64 {}

fn impls_trait<T: Trait<U>, U>() -> T {
    todo!()
}

fn foo() -> impl Sized {
    if false {
        // `opaque: Trait<u32>` shouldn't constrain `opaque` to `u32` via the blanket impl
        return impls_trait::<_, u32>();
    }
    1u64
}
fn main() {}
