// compile-flags: -Ztrait-solver=next
// revisions: works fails
//[works] check-pass

trait Trait<const N: usize> {}

impl Trait<{ 1 - 1 }> for () {}
impl Trait<{ 1 + 1 }> for () {}

fn needs<const N: usize>() where (): Trait<N> {}

#[cfg(works)]
fn main() {
    needs::<0>();
    needs::<2>();
}

#[cfg(fails)]
fn main() {
    needs::<1>();
    //[fails]~^ ERROR the trait bound `(): Trait<1>` is not satisfied
}
