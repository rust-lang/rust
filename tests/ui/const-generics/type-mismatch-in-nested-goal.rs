//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

trait A<const B: bool> {}

impl A<true> for () {}

fn needs_a<const N: usize>(_: [u8; N]) where (): A<N> {}
//~^ ERROR the constant `N` is not of type `bool`

pub fn main() {
    needs_a([]);
    //~^ ERROR the constant `true` is not of type `usize`
    //~| ERROR mismatched types
    // FIXME(const_generics): we should hide this error as we've already errored above
}
