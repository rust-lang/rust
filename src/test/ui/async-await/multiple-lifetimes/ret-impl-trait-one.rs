// edition:2018

// Test that a feature gate is needed to use `impl Trait` as the
// return type of an async.

#![feature(async_await, member_constraints)]

trait Trait<'a> { }
impl<T> Trait<'_> for T { }

// Only `'a` permitted in return type, not `'b`.
async fn async_ret_impl_trait1<'a, 'b>(a: &'a u8, b: &'b u8) -> impl Trait<'a> {
    //~^ ERROR lifetime mismatch
    (a, b)
}

// As above, but `'b: 'a`, so return type can be inferred to `(&'a u8,
// &'a u8)`.
async fn async_ret_impl_trait2<'a, 'b>(a: &'a u8, b: &'b u8) -> impl Trait<'a>
where
    'b: 'a,
{
    (a, b)
}

fn main() {
}
