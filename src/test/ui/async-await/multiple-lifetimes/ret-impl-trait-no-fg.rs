// edition:2018

// Test that a feature gate is needed to use `impl Trait` as the
// return type of an async.

trait Trait<'a, 'b> { }
impl<T> Trait<'_, '_> for T { }

async fn async_ret_impl_trait<'a, 'b>(a: &'a u8, b: &'b u8) -> impl Trait<'a, 'b> {
    //~^ ERROR ambiguous lifetime bound
    //~| ERROR ambiguous lifetime bound
    //~| ERROR ambiguous lifetime bound
    //~| ERROR hidden type for `impl Trait` captures lifetime that does not appear in bounds
    //~| ERROR hidden type for `impl Trait` captures lifetime that does not appear in bounds
    (a, b)
}

fn main() {
    let _ = async_ret_impl_trait(&22, &44);
}
