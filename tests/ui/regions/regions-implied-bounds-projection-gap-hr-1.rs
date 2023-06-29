// The "projection gap" is particularly "fun" around higher-ranked
// projections.  This is because the current code is hard-coded to say
// that a projection that contains escaping regions, like `<T as
// Trait2<'y, 'z>>::Foo` where `'z` is bound, can only be found to
// outlive a region if all components that appear free (`'y`, where)
// outlive that region. However, we DON'T add those components to the
// implied bounds set, but rather we treat projections with escaping
// regions as opaque entities, just like projections without escaping
// regions.

trait Trait1<T> { }

trait Trait2<'a, 'b> {
    type Foo;
}

// As a side-effect of the conservative process above, the type of
// this argument `t` is not automatically considered well-formed,
// since for it to be WF, we would need to know that `'y: 'x`, but we
// do not infer that.
fn callee<'x, 'y, T>(t: &'x dyn for<'z> Trait1< <T as Trait2<'y, 'z>>::Foo >)
    //~^ ERROR the trait bound `for<'z> T: Trait2<'y, 'z>` is not satisfied
{
}

fn main() { }
