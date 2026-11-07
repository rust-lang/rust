// Test for a less than ideal interaction of implied bounds and normalization.
trait Tr {
    type Ty;
}

impl<T: 'static> Tr for T {
    type Ty = &'static T;
}

// `<&'a u8 as Tr>::Ty` should cause an error because `&'a u8: Tr` doesn't hold for
// all possible 'a. However, we consider normalized types for implied bounds.
//
// We normalize this projection to `&'static &'a u8` and add a nested `&'a u8: 'static`
// bound. This bound is then proven using the implied bounds for `&'static &'a u8` which
// we only get by normalizing in the first place.
fn test<'a>(x: &'a u8, _wf: <&'a u8 as Tr>::Ty) -> &'static u8 { x }

fn main() {
    // This works as we have 'static references due to promotion.
    let _: &'static u8 = test(&3, &&3);
    // This causes an error because the projection requires 'a to be 'static.
    // It would be unsound if this compiled.
    let x: u8 = 3;
    let _: &'static u8 = test(&x, &&3);
    //~^ ERROR `x` does not live long enough

}
