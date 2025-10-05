//@ check-pass
// Test that implied bounds from type parameters are properly tracked in HRTB contexts.
// The type `&'b &'a ()` implies `'a: 'b`, and this constraint should be preserved
// when deriving supertrait bounds.

trait Subtrait<'a, 'b, R>: Supertrait<'a, 'b> {}

trait Supertrait<'a, 'b> {}

struct MyStruct;

// This implementation is valid: we only implement Supertrait for 'a: 'b
impl<'a: 'b, 'b> Supertrait<'a, 'b> for MyStruct {}

// This implementation is also valid: the type parameter &'b &'a () implies 'a: 'b
impl<'a, 'b> Subtrait<'a, 'b, &'b &'a ()> for MyStruct {}

// This function requires the HRTB on Subtrait
fn need_hrtb_subtrait<S>()
where
    S: for<'a, 'b> Subtrait<'a, 'b, &'b &'a ()>,
{
    // This should work - the bound on Subtrait with the type parameter
    // &'b &'a () implies 'a: 'b, which matches what Supertrait requires
    need_hrtb_supertrait::<S>()
}

// This function requires a weaker HRTB on Supertrait
fn need_hrtb_supertrait<S>()
where
    S: for<'a, 'b> Supertrait<'a, 'b>,
{
}

fn main() {
    need_hrtb_subtrait::<MyStruct>();
}
