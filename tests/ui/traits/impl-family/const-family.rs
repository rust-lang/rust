// A family of impls that differ only in the self type's const argument is looked up by that
// argument. Goals whose argument is not a plain value must still see every impl.
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Tr {}
struct S<const N: usize>;

macro_rules! family {
    ($tr:ident: $($n:literal)*) => { $(impl $tr for S<$n> {})* };
}
family!(Tr: 0 1 2 3 4 5 6 7);

// One impl with an unevaluated const keeps this family from being split at all.
trait Tr2 {}
family!(Tr2: 0 1 2 3);
impl Tr2 for S<{ 4 + 1 }> {}

fn requires<T: Tr>() {}
fn requires2<T: Tr2>() {}

const FIVE: usize = 5;

fn with_bound<const N: usize>()
where
    S<N>: Tr,
{
    requires::<S<N>>();
}

fn main() {
    requires::<S<5>>();
    requires::<S<{ 2 + 3 }>>();
    requires::<S<FIVE>>();
    with_bound::<7>();
    requires2::<S<5>>();
}
