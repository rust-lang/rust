//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ compile-flags: -Zpolonius=off
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

trait Trait<'a, 'b> {}
impl<'a, 'b> Trait<'a, 'b> for u8 where 'a: 'b {}
struct RequiresTrait<T: for<'a, 'b> Trait<'a, 'b>>(T);

fn main() {
    let _value = None::<RequiresTrait<u8>>;
    //[current]~^ ERROR implementation of `Trait` is not general enough
    //[next]~^^ ERROR lifetime bound not satisfied
}
