//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-pass

// Regression test for trait-system-refactor-initiative#205. Avoid
// constraining other impl arguments when applying blanket impls.

// FIXME(-Znext-solver): This currently incompletely constrains the
// argument of `opaque: Trait<?x>` using the blanket impl of trait.
// Ideally we don't do that.

trait Trait<T> {}

impl<T> Trait<u64> for T {}
impl Trait<u32> for u64 {}

fn impls_trait<T: Trait<U>, U>(_: U) -> T {
    todo!()
}

fn foo() -> impl Sized {
    let x = Default::default();
    if false {
        return impls_trait::<_, _>(x);
    }
    let _: u32 = x;
    //[next]~^ ERROR mismatched types
    1u64
}
fn main() {}
