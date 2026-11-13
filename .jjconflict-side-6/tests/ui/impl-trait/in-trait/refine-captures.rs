trait LifetimeParam<'a> {
    fn test() -> impl Sized;
}
// Refining via capturing fewer lifetimes than the trait definition.
impl<'a> LifetimeParam<'a> for i32 {
    fn test() -> impl Sized + use<> {}
    //~^ WARN impl trait in impl method captures fewer lifetimes than in trait
}
// If the lifetime is substituted, then we don't refine anything.
impl LifetimeParam<'static> for u32 {
    fn test() -> impl Sized + use<> {}
    // Ok
}

trait TypeParam<T> {
    fn test() -> impl Sized;
}
// Indirectly capturing a lifetime param through a type param substitution.
impl<'a> TypeParam<&'a ()> for i32 {
    fn test() -> impl Sized + use<> {}
    //~^ WARN impl trait in impl method captures fewer lifetimes than in trait
}
// Two of them, but only one is captured...
impl<'a, 'b> TypeParam<(&'a (), &'b ())> for u32 {
    fn test() -> impl Sized + use<'b> {}
    //~^ WARN impl trait in impl method captures fewer lifetimes than in trait
}
// What if we don't capture a type param? That should be an error otherwise.
impl<T> TypeParam<T> for u64 {
    fn test() -> impl Sized + use<> {}
    //~^ ERROR `impl Trait` must mention all type parameters in scope in `use<...>`
}

fn main() {}
