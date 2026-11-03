//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Trait {}
fn main() {
    let _ = &1_i32 as &dyn Trait;
    //[old]~^ ERROR the trait bound `i32: Trait` is not satisfied
    //[next]~^^ ERROR the trait bound `&i32: CoerceUnsized<&dyn Trait>` is not satisfied in `i32`
}
