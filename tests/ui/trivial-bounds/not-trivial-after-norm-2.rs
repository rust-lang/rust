//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait With {
    type Assoc;
}
impl With for u32 {
    type Assoc = [u32; 0];
}

trait Trait {}
impl<const N: usize> Trait for [u32; N] {}

fn foo<const N: usize>()
where
    u32: With<Assoc = [u32; N]>,
    // This where-bound is global before normalization
    // and references `T` afterwards. We check whether
    // global where-bounds hold by proving them in an empty
    // `param_env`.
    //
    // Make sure we don't introduce params by normalizing after
    // checking whether the where-bound is global. Proving
    // `[u32; N]: Trait` then caused an ICE when trying to fetch
    // the type of `N`.
    <u32 as With>::Assoc: Trait,
{
}

fn main() {}
